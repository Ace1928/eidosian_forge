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
def prior_postprocess(self, tokens):
    """
        Shifts back the input tokens if the model uses an encoder decoder architecture. As the embedding layer is
        shared, `prior_embed_dim_shift` shifts the music token ids by `lyric_vocab_size`. Only returns the music
        tokens.
        """
    batch_size = tokens.shape[0]
    dims = (self.input_shapes[0], tokens.shape[1] - self.input_shapes[0])
    tokens = list(torch.split(tokens, dims, dim=1))
    for i in range(len(tokens)):
        bins_shift = int(self.embed_dim_shift[i])
        tokens[i] = (tokens[i] - bins_shift).view(batch_size, -1)
        tokens[i] = torch.clamp(tokens[i], min=0)
    return tokens[-1]