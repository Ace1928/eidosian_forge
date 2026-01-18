import numpy.testing
import cupy
def assert_array_almost_equal_nulp(x, y, nulp=1):
    """Compare two arrays relatively to their spacing.

    Args:
         x(numpy.ndarray or cupy.ndarray): The actual object to check.
         y(numpy.ndarray or cupy.ndarray): The desired, expected object.
         nulp(int): The maximum number of unit in the last place for tolerance.

    .. seealso:: :func:`numpy.testing.assert_array_almost_equal_nulp`
    """
    numpy.testing.assert_array_almost_equal_nulp(cupy.asnumpy(x), cupy.asnumpy(y), nulp=nulp)