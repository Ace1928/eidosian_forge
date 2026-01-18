import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def split_batch(self, obj, n_samples, split_size):
    n_passes = (n_samples + split_size - 1) // split_size
    if isinstance(obj, torch.Tensor):
        return torch.split(obj, split_size, dim=0)
    elif isinstance(obj, list):
        return list(zip(*[torch.split(item, split_size, dim=0) for item in obj]))
    elif obj is None:
        return [None] * n_passes
    else:
        raise TypeError('Unknown input type')