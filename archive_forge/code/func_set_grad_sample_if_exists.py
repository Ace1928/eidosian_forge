from typing import Optional
import torch
from .expanded_weights_impl import ExpandedWeight
def set_grad_sample_if_exists(maybe_expanded_weight, per_sample_grad_fn):
    unpacked = unpack_expanded_weight_or_tensor(maybe_expanded_weight)
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        grad_sample_contribution = maybe_scale_by_batch_size(per_sample_grad_fn(unpacked), maybe_expanded_weight)
        if maybe_expanded_weight.batch_size > grad_sample_contribution.shape[0]:
            intermediate = torch.zeros(maybe_expanded_weight.batch_size, *grad_sample_contribution.shape[1:], dtype=grad_sample_contribution.dtype, device=grad_sample_contribution.device)
            intermediate[:grad_sample_contribution.shape[0]] = grad_sample_contribution
            grad_sample_contribution = intermediate
        if hasattr(unpacked, 'grad_sample') and unpacked.grad_sample is not None:
            unpacked.grad_sample = unpacked.grad_sample + grad_sample_contribution
        else:
            unpacked.grad_sample = grad_sample_contribution