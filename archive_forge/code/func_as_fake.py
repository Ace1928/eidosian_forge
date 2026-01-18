import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
def as_fake(self):
    """
        Returns a "fake" value (either a FakeTensor or a SymInt)
        representing the variable in question.  This only works
        for variables that denote Tensor or int.  You can use
        this to query metadata; e.g., v.as_fake().size(0) will
        tell you the compile-time known size of the tensor.

        WARNING: Do NOT mutate the returned tensor.
        """
    return self.__variable.as_proxy().node.meta['example_value']