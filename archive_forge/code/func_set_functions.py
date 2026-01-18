import numbers
import numpy as np
from matplotlib import _api, _docstring
import matplotlib.ticker as mticker
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator
from matplotlib.axis import Axis
def set_functions(self, functions):
    """
        Set how the secondary axis converts limits from the parent axes.

        Parameters
        ----------
        functions : 2-tuple of func, or `Transform` with an inverse.
            Transform between the parent axis values and the secondary axis
            values.

            If supplied as a 2-tuple of functions, the first function is
            the forward transform function and the second is the inverse
            transform.

            If a transform is supplied, then the transform must have an
            inverse.
        """
    if isinstance(functions, tuple) and len(functions) == 2 and callable(functions[0]) and callable(functions[1]):
        self._functions = functions
    elif functions is None:
        self._functions = (lambda x: x, lambda x: x)
    else:
        raise ValueError('functions argument of secondary axes must be a two-tuple of callable functions with the first function being the transform and the second being the inverse')
    self._set_scale()