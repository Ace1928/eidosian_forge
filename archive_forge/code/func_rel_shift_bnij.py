import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_xlnet import XLNetConfig
@staticmethod
def rel_shift_bnij(x, klen=-1):
    x_size = x.shape
    x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x = x[:, :, 1:, :]
    x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    return x