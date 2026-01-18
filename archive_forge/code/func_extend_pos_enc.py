import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
def extend_pos_enc(self, x):
    """Reset the positional encodings."""
    if self.pos_enc is not None:
        if self.pos_enc.size(1) >= x.size(1) * 2 - 1:
            if self.pos_enc.dtype != x.dtype or self.pos_enc.device != x.device:
                self.pos_enc = self.pos_enc.to(dtype=x.dtype, device=x.device)
            return
    pos_enc_positive = torch.zeros(x.size(1), self.embed_dim)
    pos_enc_negative = torch.zeros(x.size(1), self.embed_dim)
    position = torch.arange(0, x.size(1), dtype=torch.int64).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.int64).float() * -(math.log(10000.0) / self.embed_dim))
    pos_enc_positive[:, 0::2] = torch.sin(position * div_term)
    pos_enc_positive[:, 1::2] = torch.cos(position * div_term)
    pos_enc_negative[:, 0::2] = torch.sin(-1 * position * div_term)
    pos_enc_negative[:, 1::2] = torch.cos(-1 * position * div_term)
    pos_enc_positive = torch.flip(pos_enc_positive, [0]).unsqueeze(0)
    pos_enc_negative = pos_enc_negative[1:].unsqueeze(0)
    pos_enc = torch.cat([pos_enc_positive, pos_enc_negative], dim=1)
    self.pos_enc = pos_enc.to(device=x.device, dtype=x.dtype)