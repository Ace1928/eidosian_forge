import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def encode_text(self, text):
    if text.ndim is None:
        raise ValueError('text must not be NoneType')
    if text.ndim not in [2, 3]:
        raise ValueError('Number of dimensions in text must be 2 or 3')
    squeeze_dim = False
    num_text = 1
    if text.ndim == 3:
        num_text = text.shape[1]
        batch_size, num_text, hidden_dim = text.shape
        text = text.reshape(batch_size * num_text, hidden_dim)
        squeeze_dim = True
    encoded_text = self.text_encoder(text)
    text_queries = self.text_projector(encoded_text)
    if squeeze_dim:
        _, hidden_dim = text_queries.shape
        text_queries = text_queries.reshape(batch_size, num_text, hidden_dim)
        if self.prompt_ctx is not None:
            text_queries_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_queries.shape[0], 1, 1)
            text_queries = torch.cat([text_queries, text_queries_ctx], dim=1)
    return text_queries