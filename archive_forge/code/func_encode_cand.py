import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
def encode_cand(self, words):
    """
        Encode the candidates.
        """
    if words is None:
        return None
    if words.dim() == 3:
        oldshape = words.shape
        words = words.reshape(oldshape[0] * oldshape[1], oldshape[2])
    else:
        oldshape = None
    encoded = self.cand_encoder(words)
    if oldshape is not None:
        encoded = encoded.reshape(oldshape[0], oldshape[1], -1)
    return encoded