import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(self, input2):
    if config.coordinate_descent_tuning:
        if self.shape[0] == 1 or input2.shape[1] == 1:
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    if self.device.type == 'cpu':
        if self.size(-1) == 1 and self.size(0) > 0 and (input2.size(0) == 1) and (self.dtype == input2.dtype) and (torch.numel(self) + torch.numel(input2) <= 32):
            return torch.cat([self[i, :] * input2 for i in range(self.size(0))])
        if self.size(0) == 1 and input2.size(-1) == 1:
            return torch.sum(self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True).unsqueeze(0)
    return NotImplemented