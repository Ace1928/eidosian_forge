import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu_new, silu
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
from .configuration_openai import OpenAIGPTConfig
def split_heads(self, x, k=False):
    new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
    x = x.view(*new_x_shape)
    if k:
        return x.permute(0, 2, 3, 1)
    else:
        return x.permute(0, 2, 1, 3)