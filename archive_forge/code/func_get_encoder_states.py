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
def get_encoder_states(self, lyric_tokens, sample=False):
    """
        Retreive the last hidden_states of the lyric encoder that will be attended to by the decoder. Forwards through
        the lyric encoder.
        """
    if self.nb_relevant_lyric_tokens != 0 and self.lyric_conditioning:
        if sample:
            self.encoder = self.encoder.to(lyric_tokens.device)
        lyric_acts = self.encoder(lyric_tokens, None, None, None)
        lyric_acts = self.encoder.proj_in(lyric_acts)
        last_encoder_hidden_states = self.encoder.final_layer_norm(lyric_acts)
    else:
        last_encoder_hidden_states = None
    return last_encoder_hidden_states