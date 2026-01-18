import numpy
import modin.numpy as np
def assert_scalar_or_array_equal(x1, x2, err_msg=''):
    """
    Assert whether the result of the numpy and modin computations are the same.

    If either argument is a modin array object, then `_to_numpy()` is called on it.
    The arguments are compared with `numpy.testing.assert_array_equals`.
    """
    if isinstance(x1, np.array):
        x1 = x1._to_numpy()
    if isinstance(x2, np.array):
        x2 = x2._to_numpy()
    numpy.testing.assert_array_equal(x1, x2, err_msg=err_msg)