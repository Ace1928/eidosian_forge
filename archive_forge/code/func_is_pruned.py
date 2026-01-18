import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def is_pruned(module):
    """Check if a module is pruned by looking for pruning pre-hooks.

    Check whether ``module`` is pruned by looking for
    ``forward_pre_hooks`` in its modules that inherit from the
    :class:`BasePruningMethod`.

    Args:
        module (nn.Module): object that is either pruned or unpruned

    Returns:
        binary answer to whether ``module`` is pruned.

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = nn.Linear(5, 7)
        >>> print(prune.is_pruned(m))
        False
        >>> prune.random_unstructured(m, name='weight', amount=0.2)
        >>> print(prune.is_pruned(m))
        True
    """
    for _, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            if isinstance(hook, BasePruningMethod):
                return True
    return False