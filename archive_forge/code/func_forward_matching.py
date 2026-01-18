from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def forward_matching(self, x: Tensor) -> Tensor:
    logits = self.matching_head(x)
    if self.head_one_neuron:
        return torch.sigmoid(logits)[:, 0]
    return F.softmax(logits, dim=1)[:, 1]