from functools import update_wrapper
from numbers import Number
from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch.overrides import is_tensor_like
def probs_to_logits(probs, is_binary=False):
    """
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)