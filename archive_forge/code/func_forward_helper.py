from typing import Optional
import torch
from .expanded_weights_impl import ExpandedWeight
def forward_helper(func, expanded_args, expanded_kwargs):
    """Compute the forward pass for a function that has expanded weight(s) passed to it.

    It will run the forward pass where all ExpandedWeights are their original
    weight. It runs checks on the given arguments and detaches the outputs.

    .. note:: First argument in :attr:`expanded_args` must be the input with the batch
    dimension as the first element of the shape

    .. note:: :attr:`func` must return a Tensor or tuple of Tensors

    Args:
        func: The function to be called
        expanded_args: Arguments to be passed to :attr:`func`. Will include arguments
          that need to be unpacked because they are ExpandedWeights
        expanded_kwargs: Keyword arguments to be passed to :attr:`func`.
          Similar to :attr:`expanded_args`.
    """
    unexpanded_args, unexpanded_kwargs = _check_and_unexpand_args(func, expanded_args, expanded_kwargs)
    return func(*unexpanded_args, **unexpanded_kwargs)