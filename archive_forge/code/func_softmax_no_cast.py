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
@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of type bfloat16
    """
    d = t.dtype
    if d is torch.bfloat16 and (not is_deepspeed_initialized()):
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)
    return s