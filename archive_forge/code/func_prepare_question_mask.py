import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
@staticmethod
def prepare_question_mask(q_lengths: torch.Tensor, maxlen: int):
    mask = torch.arange(0, maxlen).to(q_lengths.device)
    mask.unsqueeze_(0)
    mask = torch.where(mask < q_lengths, 1, 0)
    return mask