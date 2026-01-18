import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_rwkv import RwkvConfig
def extract_key_value(self, hidden, state=None):
    if hidden.size(1) == 1 and state is not None:
        shifted = state[1][:, :, self.layer_id]
    else:
        shifted = self.time_shift(hidden)
        if state is not None:
            shifted[:, 0] = state[1][:, :, self.layer_id]
    key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
    value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
    receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
    key = self.key(key)
    value = self.value(value)
    receptance = torch.sigmoid(self.receptance(receptance))
    if state is not None:
        state[1][:, :, self.layer_id] = hidden[:, -1]
    return (receptance, key, value, state)