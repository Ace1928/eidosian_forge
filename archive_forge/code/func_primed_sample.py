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
@add_start_docstrings('Generate a raw audio conditioned on the provided `raw_audio` which is used as conditioning at each of the\n        generation levels. The audio is encoded to music tokens using the 3 levels of the VQ-VAE. These tokens are\n        used: as conditioning for each level, which means that no ancestral sampling is required.\n\n        Args:\n            raw_audio (`List[torch.Tensor]` of length `n_samples` ) :\n                A list of raw audio that will be used as conditioning information for each samples that will be\n                generated.\n        ', JUKEBOX_SAMPLING_INPUT_DOCSTRING)
def primed_sample(self, raw_audio, labels, **sampling_kwargs) -> List[torch.LongTensor]:
    sample_levels = sampling_kwargs.pop('sample_levels', list(range(len(self.priors))))
    self.vqvae.to(raw_audio.device).float()
    with torch.no_grad():
        music_tokens = self.vqvae.encode(raw_audio, start_level=0, end_level=len(self.priors), bs_chunks=raw_audio.shape[0])
    music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
    return music_tokens