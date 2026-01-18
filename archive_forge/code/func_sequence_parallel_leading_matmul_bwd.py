from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
@make_pytorch_cuda_operator
def sequence_parallel_leading_matmul_bwd(scattered_input: torch.Tensor, weights: List[torch.Tensor], grad_gathered_outputs: List[torch.Tensor], fuse: bool, process_group: torch.distributed.ProcessGroup) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    mp_size = process_group.size()
    if fuse:
        grad_scattered_input = torch.empty_like(scattered_input)
        grad_weights = [torch.zeros_like(w) for w in weights]
        grad_gathered_outputss = [grad_go.tensor_split(mp_size, dim=0) for grad_go in grad_gathered_outputs]

        def my_si_matmul(grad_gathered_inputs: List[torch.Tensor], dst_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
            grad_gi, = grad_gathered_inputs
            with torch.cuda.stream(stream_factory()):
                tiled_matmul_fwd([[grad_gos[dst_rank] for grad_gos in grad_gathered_outputss]], [[w.t()] for w in weights], out=[[grad_gi]])
        fused_anything_and_reducescatter(my_si_matmul, [grad_scattered_input], group=process_group)
        events = [torch.cuda.Event() for _ in weights]

        def my_w_matmul(gathered_inputs_shard: List[torch.Tensor], src_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
            gi_shard, = gathered_inputs_shard
            for grad_gos, grad_w, event in zip(grad_gathered_outputss, grad_weights, events):
                with torch.cuda.stream(stream_factory()):
                    event.wait()
                    grad_w.t().addmm_(grad_gos[src_rank].t(), gi_shard)
                    event.record()
        fused_allgather_and_anything([scattered_input], my_w_matmul, group=process_group)
    else:
        gathered_input, handle = gather_along_first_dim_async(scattered_input, process_group=process_group)
        (grad_gathered_input,), = tiled_matmul_fwd([[grad_go for grad_go in grad_gathered_outputs]], [[w.t()] for w in weights])
        if handle is not None:
            handle.wait()
        grad_scattered_input, handle = reduce_scatter_along_first_dim_async(grad_gathered_input, process_group=process_group)
        grad_weights_tuples = tiled_matmul_fwd([[grad_go.t()] for grad_go in grad_gathered_outputs], [[gathered_input]])
        if handle is not None:
            handle.wait()
        grad_weights = [grad_w.t() for grad_w, in grad_weights_tuples]
    return (grad_scattered_input, grad_weights)