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
def sample_partial_window(self, music_tokens, labels, offset, sampling_kwargs, level, tokens_to_sample, max_batch_size):
    prior = self.priors[level]
    sampled_tokens = music_tokens[level]
    n_ctx = prior.n_ctx
    nb_sampled_tokens = sampled_tokens.shape[1]
    if nb_sampled_tokens < n_ctx - tokens_to_sample:
        sampling_kwargs['sample_tokens'] = nb_sampled_tokens + tokens_to_sample
        start = 0
    else:
        sampling_kwargs['sample_tokens'] = n_ctx
        start = nb_sampled_tokens - n_ctx + tokens_to_sample
    return self.sample_single_window(music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size)