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
def get_emb(self, sample_t, n_samples, tokens, audio_conditioning, metadata_conditioning):
    if sample_t == 0:
        hidden_states = torch.empty(n_samples, 1, self.width, dtype=self.embed_tokens.weight.dtype).to(self.embed_tokens.weight.device)
        if self.metadata_conditioning:
            hidden_states[:, 0] = metadata_conditioning.view(n_samples, self.width)
        else:
            hidden_states[:, 0] = self.start_token
    else:
        hidden_states = self.embed_tokens(tokens)
    if audio_conditioning.shape == (n_samples, self.n_ctx, self.width):
        cond = audio_conditioning[:, sample_t:sample_t + 1, :]
    else:
        cond = audio_conditioning
    hidden_states = hidden_states + self.pos_emb()[sample_t:sample_t + 1] + cond
    return (hidden_states, cond)