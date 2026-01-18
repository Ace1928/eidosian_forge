from typing import Optional
import torch
from .expanded_weights_impl import ExpandedWeight
def is_batch_first(expanded_args_and_kwargs):
    batch_first = None
    for arg in expanded_args_and_kwargs:
        if not isinstance(arg, ExpandedWeight):
            continue
        if not batch_first:
            batch_first = arg.batch_first
        elif arg.batch_first != batch_first:
            raise RuntimeError('Got conflicting batch_first arguments in the same layer')
    return batch_first