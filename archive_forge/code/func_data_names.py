import logging
import copy
from ..initializer import Uniform
from .base_module import BaseModule
@property
def data_names(self):
    """A list of names for data required by this module."""
    if len(self._modules) > 0:
        return self._modules[0].data_names
    return []