import logging
import copy
from ..initializer import Uniform
from .base_module import BaseModule
@property
def data_shapes(self):
    """Gets data shapes.

        Returns
        -------
        list
            A list of `(name, shape)` pairs. The data shapes of the first module
            is the data shape of a `SequentialModule`.
        """
    assert self.binded
    return self._modules[0].data_shapes