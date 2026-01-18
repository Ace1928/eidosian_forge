import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
def flip_z_cache_(z_cache, z):
    quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
    z_cache = z_cache.transpose(row_dim, col_dim)
    z_cache = z_cache[..., :n // 2, :, :]
    first_half_slicer = empty_slicer(z_cache)
    first_half_slicer[col_dim] = slice(0, half_n)
    z_cache[first_half_slicer] = quadrant_3
    quadrant_4 = slice_tensor(z, half_n, None, row_dim)
    quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)
    quadrant_3_slicer = empty_slicer(z_cache)
    quadrant_3_slicer[col_dim] = slice(half_n, None)
    z_cache[quadrant_3_slicer] = quadrant_4
    return z_cache