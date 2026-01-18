import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def get_as_parameter(self):
    """Deprecated getter for the `_ctypes._as_parameter_` property.

        .. deprecated:: 1.21
        """
    warnings.warn('"get_as_parameter" is deprecated. Use "_as_parameter_" instead', DeprecationWarning, stacklevel=2)
    return self._as_parameter_