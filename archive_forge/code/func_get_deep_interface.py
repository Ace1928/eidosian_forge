import warnings
import autoray as ar
import numpy as _np
from autograd.numpy.numpy_boxes import ArrayBox
from autoray import numpy as np
from . import single_dispatch  # pylint:disable=unused-import
def get_deep_interface(value):
    """
    Given a deep data structure with interface-specific scalars at the bottom, return their
    interface name.

    Args:
        value (list, tuple): A deep list-of-lists, tuple-of-tuples, or combination with
            interface-specific data hidden within it

    Returns:
        str: The name of the interface deep within the value

    **Example**

    >>> x = [[jax.numpy.array(1), jax.numpy.array(2)], [jax.numpy.array(3), jax.numpy.array(4)]]
    >>> get_deep_interface(x)
    'jax'

    This can be especially useful when converting to the appropriate interface:

    >>> qml.math.asarray(x, like=qml.math.get_deep_interface(x))
    Array([[1, 2],
       [3, 4]], dtype=int64)

    """
    itr = value
    while isinstance(itr, (list, tuple)):
        if len(itr) == 0:
            return 'builtins'
        itr = itr[0]
    return ar.infer_backend(itr)