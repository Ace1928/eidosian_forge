import os
from collections import namedtuple
from typing import Any
import torch
from .grad_mode import _DecoratorContextManager
def enter_dual_level():
    """Enter a new forward grad level.

    This level can be used to make and unpack dual Tensors to compute
    forward gradients.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    global _current_level
    new_level = torch._C._enter_dual_level()
    if new_level != _current_level + 1:
        raise RuntimeError('Entering a new forward AD level but the current level is not valid. Make sure you did not modified it directly.')
    _current_level = new_level
    return new_level