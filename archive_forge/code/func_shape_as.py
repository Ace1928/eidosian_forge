import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def shape_as(self, obj):
    """
        Return the shape tuple as an array of some other c-types
        type. For example: ``self.shape_as(ctypes.c_short)``.
        """
    if self._zerod:
        return None
    return (obj * self._arr.ndim)(*self._arr.shape)