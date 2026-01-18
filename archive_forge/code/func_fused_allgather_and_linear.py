import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def fused_allgather_and_linear(scattered_input: torch.Tensor, weight: Union[torch.Tensor, List[torch.Tensor]], *, group: dist.ProcessGroup, out: Optional[Union[torch.Tensor, List[torch.Tensor]]]=None, num_stripes: int=1, timeout_s: int=60 * 60, scale_scattered_input: Optional[torch.Tensor]=None, scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]=None, out_dtype: Optional[torch.dtype]=None, **private_args_DO_NOT_USE) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Performs a fused all-gather followed by a linear op

    It is equivalent to the following plain PyTorch code:

    # like scattered_input but with first dim multiplied by group's world size
    gathered_input = scattered_input.new_empty(...)
    dist.all_gather_into_tensor(gathered_input, scattered_input, group=group)
    return torch.nn.functional.linear(gathered_input, weight)

    It achieves this by breaking down the matmul into smaller partial ops (as
    many as the world size), each needing as input a different "contribution"
    to the all-gather (by a different rank), and writing to a different chunk of
    the output. Then, on one stream, it sends the local contribution to all
    other ranks (first one rank over, then two, ...) while, on another stream,
    it launches the sub-matmuls in the order in which the remote contributions
    (which are the sub-matmuls' inputs) are supposed to arrive, so that ideally
    none of the sub-matmuls will ever have to wait.

    The idea comes from this paper: https://arxiv.org/abs/2302.05442

    This method uses a staging buffer, which persists across calls, of the same
    size as the all-gathered input tensor (i.e., the input's size times the
    world size). If multiple inputs of multiple sizes are used, the staging
    buffer will be the maximum needed by any of them. Each call, when it starts,
    must first wait for the previous call to finish using the staging buffer. In
    normal conditions, where there's some other operation between two calls,
    this isn't an issue. However, when doing back-to-back calls (like in
    benchmarks) it can introduce artificial delays. To hide them, we allow using
    more than one staging buffer, which will be cycled through, thus trading
    memory for speed. This can be controlled using the num_stripes argument.

    Supports FP8 gemm for tensor-wise quantized weight and input tensors.
    To enable FP8 gemm:
    1. pass scattered_input and weight as quantized FP8 datatype
    2. pass scale_scattered_input and scale_weight, the scales used to
    quantize input and weight, respectively.
    3. set out_dtype, if not specified, will be inferred from scattered_input type.

    """
    world_size = group.size()
    weights = weight if isinstance(weight, list) else [weight]
    assert (scale_scattered_input is None) == (scale_weight is None)
    if scale_weight is not None:
        assert isinstance(weight, list) == isinstance(scale_weight, list)
        scales_weights = scale_weight if isinstance(scale_weight, list) else [scale_weight]
        assert len(weights) == len(scales_weights)
        assert out_dtype is not None, 'output_dtype is required with FP8'
    else:
        scales_weights = [torch.empty(1)] * len(weights)
    assert all((w.ndim == 2 for w in weights))
    assert scattered_input.ndim >= 2
    assert all((scattered_input.shape[-1] == w.shape[-1] for w in weights))
    assert scattered_input.is_contiguous()
    gathered_input_shape = (world_size,) + scattered_input.shape
    gathered_output_shapes = [gathered_input_shape[:-1] + w.shape[:-1] for w in weights]
    if out is not None:
        assert isinstance(out, list) == isinstance(weight, list)
        gathered_outputs = out if isinstance(out, list) else [out]
        assert len(gathered_outputs) == len(gathered_output_shapes)
        assert all((go.shape == gos for go, gos in zip(gathered_outputs, gathered_output_shapes)))
        assert all((go.is_contiguous() for go in gathered_outputs))
        if out_dtype is not None:
            if isinstance(out, list):
                for o in out:
                    assert o.dtype == out_dtype
            else:
                assert out.dtype == out_dtype
    else:
        gathered_outputs = [scattered_input.new_empty(gos, dtype=out_dtype if out_dtype is not None else scattered_input.dtype) for gos in gathered_output_shapes]

    def my_matmul(inputs: List[torch.Tensor], src_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
        for w, scale_weight, go in zip(weights, scales_weights, gathered_outputs):
            with torch.cuda.stream(stream_factory()):
                if _is_fp8_dtype(w.dtype):
                    output_amax = torch.empty(1, dtype=torch.float32, device=w.device)
                    torch._scaled_mm(inputs[0], w.t(), out_dtype=go[src_rank].dtype, scale_a=scale_scattered_input, scale_b=scale_weight, out=(go[src_rank], output_amax))
                else:
                    torch.matmul(inputs[0], w.t(), out=go[src_rank])
    _is_regular_matmul = all([not _is_fp8_dtype(w.dtype) for w in weights])
    fused_allgather_and_anything([scattered_input], my_matmul, group=group, num_stripes=num_stripes, timeout_s=timeout_s, _is_regular_matmul=_is_regular_matmul, _extra_triton_args=dict(bs=[w.t() for w in weights], cs=[go.flatten(0, -2) for go in gathered_outputs], cs_my_shard=None), **private_args_DO_NOT_USE)
    if isinstance(weight, list):
        return [go.flatten(0, 1) for go in gathered_outputs]
    else:
        return gathered_outputs[0].flatten(0, 1)