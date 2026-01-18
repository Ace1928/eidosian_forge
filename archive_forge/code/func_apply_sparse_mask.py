import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
    return attn_weights