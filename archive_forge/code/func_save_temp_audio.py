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
def save_temp_audio(fname, lvl, metas, aud):
    aud = torch.clamp(aud, -1, 1).cpu().numpy()
    for i in list(range(aud.shape[0])):
        if metas is not None:
            artists, genres, lyrics = list(metas)[i].values()
            path = f'{fname}/lvl_{lvl}-{artists}-{genres}-{lyrics[:5]}-{i}'
            np.save(path, aud[i])
        else:
            np.save(f'{fname}/lvl_{lvl}-sample-{i}', aud[i])