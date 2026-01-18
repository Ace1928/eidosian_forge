import itertools
from collections import OrderedDict
import numpy as np
def possibly_convert_to_numpy(x):
    """Convert things without a 'shape' to ndarrays, but leave everything else.

    Examples
    --------
    >>> oe.parser.possibly_convert_to_numpy(5)
    array(5)

    >>> oe.parser.possibly_convert_to_numpy([5, 3])
    array([5, 3])

    >>> oe.parser.possibly_convert_to_numpy(np.array([5, 3]))
    array([5, 3])

    # Any class with a shape is passed through
    >>> class Shape:
    ...     def __init__(self, shape):
    ...         self.shape = shape
    ...

    >>> myshape = Shape((5, 5))
    >>> oe.parser.possibly_convert_to_numpy(myshape)
    <__main__.Shape object at 0x10f850710>
    """
    if not hasattr(x, 'shape'):
        return np.asanyarray(x)
    else:
        return x