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
def collate_dense_tensors(samples: List[torch.Tensor], pad_v: float=0) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11, ..., d_1K),
         (d_21, ..., d_2K), ..., (d_N1, ..., d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len({x.dim() for x in samples}) != 1:
        raise RuntimeError(f'Samples has varying dimensions: {[x.dim() for x in samples]}')
    device, = tuple({x.device for x in samples})
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple((slice(0, k) for k in t.shape))] = t
    return result