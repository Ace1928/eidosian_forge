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
@register_decomposition(aten.masked_scatter)
def masked_scatter(self, mask, source):
    if self.device.type == 'cuda':
        self, mask = aten.broadcast_tensors([self, mask])
        source_idx = mask.reshape(-1).cumsum(0) - 1
        return inductor_prims.masked_scatter_with_index(self, mask, source_idx, source)
    return NotImplemented