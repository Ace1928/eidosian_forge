from typing import Optional
import torch
from .expanded_weights_impl import ExpandedWeight
def maybe_scale_by_batch_size(grad_sample, expanded_weight):
    if expanded_weight.loss_reduction == 'mean':
        return grad_sample * expanded_weight.batch_size
    else:
        return grad_sample