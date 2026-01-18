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
def sample_level(self, music_tokens, labels, offset, sampling_kwargs, level, total_length, hop_length, max_batch_size):
    if total_length >= self.priors[level].n_ctx:
        iterator = get_starts(total_length, self.priors[level].n_ctx, hop_length)
        for start in iterator:
            music_tokens = self.sample_single_window(music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size)
    else:
        music_tokens = self.sample_partial_window(music_tokens, labels, offset, sampling_kwargs, level, total_length, max_batch_size)
    return music_tokens