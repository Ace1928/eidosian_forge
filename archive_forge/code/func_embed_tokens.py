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
def embed_tokens(self, music_tokens_conds):
    """
        Embeds the upper level music tokens and upsamples them to provide as audio conditioning.
        """
    music_tokens_conds = music_tokens_conds[:self.cond_level + 1]
    audio_conditioning = None
    for music_tokens_cond, conditioner_block in reversed(list(zip(music_tokens_conds, [self.conditioner_blocks]))):
        audio_conditioning = conditioner_block(music_tokens_cond, audio_conditioning)
    return audio_conditioning