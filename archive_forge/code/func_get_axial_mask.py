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
def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used in row column attentions.

    Input:
      mask: B x L tensor of booleans

    Output:
      mask: B x L x L tensor of booleans
    """
    if mask is None:
        return None
    if len(mask.shape) != 2:
        raise ValueError(f'`mask` should be a 2d-tensor, got {len(mask.shape)} dims.')
    batch_dim, seq_dim = mask.shape
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m