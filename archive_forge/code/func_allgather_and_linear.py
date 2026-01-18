import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def allgather_and_linear(self, scattered_inputs: List[torch.Tensor], my_matmul: Callable[[List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None], timeout_s: int, _wait: bool=True, _memcpy: bool=True, _triton: bool=True, _is_regular_matmul: bool=False, _extra_triton_args: Mapping[str, Any]={}):
    """Perform a fused all-gather followed by a linear layer"""
    assert all((si.device == self.my_device for si in scattered_inputs))
    assert all((si.dtype == self.dtype for si in scattered_inputs))
    scattered_input_numels = [si.numel() for si in scattered_inputs]
    total_scattered_input_numel = sum(scattered_input_numels)
    self._ensure_staging_is_large_enough(total_scattered_input_numel, random_init=_memcpy is False)
    stripe = self.next_stripe % self.num_stripes
    self.next_stripe += 1
    seq_num = self.next_seq_nums[stripe] % SEQ_NUM_WRAP_AROUND
    prev_seq_num = (seq_num - 1) % SEQ_NUM_WRAP_AROUND
    self.next_seq_nums[stripe] += 1
    stagings = [s.view((self.world_size,) + si.shape) for s, si in zip(self.staging[stripe, :, :total_scattered_input_numel].split(scattered_input_numels, dim=-1), scattered_inputs)]
    buddys_stagings = [[bs] * len(scattered_inputs) if bs.numel() == 0 else [s.view(si.shape) for s, si in zip(bs[stripe, :total_scattered_input_numel].split(scattered_input_numels, dim=-1), scattered_inputs)] for bs in self.buddys_staging]
    current_stream = torch.cuda.current_stream()
    self.memcpy_stream.wait_stream(current_stream)
    if _wait:
        WaitValues.OPERATOR([self.num_reads_from_buddys_staging[(self.my_rank + iter_) % self.world_size, stripe] for iter_ in range(1, self.world_size)], prev_seq_num, self.memcpy_stream, timeout_s)
    for iter_ in range(1, self.world_size):
        dst_rank = (self.my_rank + iter_) % self.world_size
        if _memcpy:
            with torch.cuda.stream(self.memcpy_stream):
                for bs, si in zip(buddys_stagings[dst_rank], scattered_inputs):
                    bs.copy_(si)
        self.write_stream.wait_stream(self.memcpy_stream)
        if _wait:
            Memset32bAsync.OPERATOR(self.num_writes_into_buddys_staging[dst_rank][stripe], seq_num, self.write_stream)
    if _is_regular_matmul and self._should_use_triton(_triton):
        _launch_triton_matmul(a_my_shard=scattered_inputs[0].flatten(0, -2), a=stagings[0].flatten(0, -2), my_rank=self.my_rank, world_size=self.world_size, wait_counters=self.num_writes_into_my_staging, write_counters=None, direction=BACKWARDS_WITH_ME_FIRST, stripe=stripe, seq_num=seq_num, num_stripes=self.num_stripes, timeout_s=timeout_s, _wait=_wait, **_extra_triton_args)
    else:
        self.wait_stream.wait_stream(current_stream)
        self.second_stream.wait_stream(current_stream)
        stream_factory = self.make_stream_factory(current_stream)
        my_matmul(scattered_inputs, self.my_rank, stream_factory)
        for iter_ in range(1, self.world_size):
            src_rank = (self.my_rank - iter_) % self.world_size
            if _wait:
                WaitValues.OPERATOR([self.num_writes_into_my_staging[src_rank, stripe]], seq_num, self.wait_stream, timeout_s)
                current_stream.wait_stream(self.wait_stream)
                self.second_stream.wait_stream(self.wait_stream)
            my_matmul([s[src_rank] for s in stagings], src_rank, stream_factory)
        current_stream.wait_stream(self.second_stream)
    self.write_stream.wait_stream(current_stream)
    if _wait:
        WriteValues.OPERATOR([self.num_reads_from_my_staging[(self.my_rank - iter_) % self.world_size][stripe] for iter_ in range(1, self.world_size)], seq_num, self.write_stream)