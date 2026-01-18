import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def get_dilation(i):
    return dilation[i] if isinstance(dilation, tuple) else dilation