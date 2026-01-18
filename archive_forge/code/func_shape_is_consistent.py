from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Sequence, Union
import numpy as np
def shape_is_consistent(self, prop, value) -> bool:
    """Return whether shape of values is consistent with properties.

        For example, forces of shape (7, 3) are consistent
        unless properties already have "natoms" with non-7 value.
        """
    shapespec = prop.shapespec
    shape = np.shape(value)
    if len(shapespec) != len(shape):
        return False
    for dimspec, dim in zip(shapespec, shape):
        if isinstance(dimspec, str):
            dimspec = self._dct.get(dimspec, dim)
        if dimspec != dim:
            return False
    return True