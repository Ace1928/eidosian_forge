import math
from typing import Any, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
def make_kron(w1, w2, scale=1.0):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    return rebuild * scale