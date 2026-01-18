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
def bitcast_u8_to_f32(u8):
    x, y, z, w = (u8[..., n].to(torch.int32) for n in (0, 1, 2, 3))
    if sys.byteorder == 'little':
        return (x + (y << 8) + (z << 16) + (w << 24)).view(torch.float32)[..., None]
    else:
        return ((x << 24) + (y << 16) + (z << 8) + w).view(torch.float32)[..., None]