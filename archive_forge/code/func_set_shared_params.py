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
def set_shared_params(self, model_config):
    """
        Initialises the parameters that are shared. This has to be done here because the list of `JukeboxPriorConfig`
        is nest, and is thus unreachable in the `from_dict` function
        """
    for config in model_config.prior_configs:
        config.sampling_rate = model_config.sampling_rate
        config.timing_dims = model_config.timing_dims
        config.min_duration = model_config.min_duration
        config.max_duration = model_config.max_duration
        config.max_nb_genres = model_config.max_nb_genres
        config.metadata_conditioning = model_config.metadata_conditioning