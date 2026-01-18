import torch
from .fmha import (
from .indexing import index_select_cat, scaled_index_add
from .ipc import init_ipc
from .modpar_layers import ColumnParallelLinear, RowParallelLinear
from .rmsnorm import RMSNorm
from .rope_padded import rope_padded
from .seqpar import sequence_parallel_leading_matmul, sequence_parallel_trailing_matmul
from .sequence_parallel_fused_ops import (
from .sp24 import Sparse24Tensor, sparsify24, sparsify24_like
from .swiglu_op import (
from .tiled_matmul import tiled_matmul
from .unbind import get_stack_strides, stack_or_none, unbind
def masked_matmul(a, b, mask=None):
    if torch.overrides.has_torch_function((a, b, mask)):
        return torch.overrides.handle_torch_function(masked_matmul, (a, b, mask), a, b, mask)
    att = a @ b
    if mask is None:
        return att
    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        att[~mask] = float('-inf')
    else:
        att += mask
    return att