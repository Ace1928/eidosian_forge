import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def custom_from_mask(module, name, mask):
    """Prune tensor corresponding to parameter called ``name`` in ``module`` by applying the pre-computed mask in ``mask``.

    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.custom_from_mask(
        ...     nn.Linear(5, 3), name='bias', mask=torch.tensor([0, 1, 0])
        ... )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])

    """
    CustomFromMask.apply(module, name, mask)
    return module