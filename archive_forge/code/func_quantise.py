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
def quantise(self, latent_states):
    codebook_weights = self.codebook.t()
    distance = torch.sum(latent_states ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(latent_states, codebook_weights) + torch.sum(codebook_weights ** 2, dim=0, keepdim=True)
    min_distance, music_tokens = torch.min(distance, dim=-1)
    fit = torch.mean(min_distance)
    return (music_tokens, fit)