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
def compute_projection_helper(pair, mask, a=True):
    if a:
        linear_g = self.linear_a_g
        linear_p = self.linear_a_p
    else:
        linear_g = self.linear_b_g
        linear_p = self.linear_b_p
    pair = self.layer_norm_in(pair)
    p = linear_g(pair)
    p.sigmoid_()
    p *= linear_p(pair)
    p *= mask
    p = permute_final_dims(p, (2, 0, 1))
    return p