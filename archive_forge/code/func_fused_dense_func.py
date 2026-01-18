from functools import partial
from typing import Optional
import fused_dense_lib as fused_dense_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup
from flash_attn.ops.activations import gelu_bwd, relu_bwd, sqrelu_bwd, sqrelu_fwd
from flash_attn.utils.distributed import (
def fused_dense_func(x: Tensor, weight: Tensor, bias: Optional[Tensor]=None, return_residual: bool=False, process_group: Optional[ProcessGroup]=None, sequence_parallel: bool=True):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (x.dtype == torch.float32 and torch.is_autocast_enabled())
    if x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda) and dtype_eligible:
        return FusedDenseFunc.apply(x, weight, bias, return_residual, process_group, sequence_parallel)
    else:
        assert process_group is None
        out = F.linear(x, weight, bias)
        return out if not return_residual else (out, x)