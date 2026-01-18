import logging
from typing import Union
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
def quantize_tensor(self, weight):
    max_abs = torch.abs(weight).max()
    weight_normed = weight / max_abs
    weight_normed_expanded = weight_normed.unsqueeze(-1)
    L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)
    abs_diff = torch.abs(weight_normed_expanded - L_reshaped)
    qweight = torch.argmin(abs_diff, dim=-1)
    return (qweight, max_abs)