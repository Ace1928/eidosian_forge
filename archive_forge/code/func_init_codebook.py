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
def init_codebook(self, hidden_states):
    nb_discrete_codes = self.nb_discrete_codes
    self.init = True
    codes = self._tile(hidden_states)
    self.codebook = codes[torch.randperm(codes.shape[0])][:nb_discrete_codes]
    self.codebook_sum = self.codebook
    self.codebook_elem = torch.ones(nb_discrete_codes, device=self.codebook.device)