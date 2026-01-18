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
def get_metadata(self, labels, start, total_length, offset, get_indices=False):
    metadata = labels.clone()
    metadata[:, 0] = total_length
    metadata[:, 2] = int(self.sample_length)
    metadata[:, 1:2] = int(offset * self.raw_to_tokens) + int(start * self.raw_to_tokens)
    metadata, indices = self.set_metadata_lyric_tokens(metadata)
    if get_indices:
        return (metadata, indices)
    else:
        return metadata