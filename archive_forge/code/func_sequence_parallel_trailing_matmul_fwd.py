from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
@make_pytorch_cuda_operator
def sequence_parallel_trailing_matmul_fwd(gathered_input: torch.Tensor, weight: torch.Tensor, fuse: bool, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    if fuse:
        scattered_output = fused_linear_and_reducescatter(gathered_input, weight.t(), group=process_group)
    else:
        gathered_output = torch.matmul(gathered_input, weight)
        scattered_output = reduce_scatter_along_first_dim(gathered_output, process_group=process_group)
    return scattered_output