import numpy.testing
import cupy
def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """Check that all items of arrays differ in at most N Units in the Last Place.

    Args:
         a(numpy.ndarray or cupy.ndarray): The actual object to check.
         b(numpy.ndarray or cupy.ndarray): The desired, expected object.
         maxulp(int): The maximum number of units in the last place
             that elements of ``a`` and ``b`` can differ.
         dtype(numpy.dtype): Data-type to convert ``a`` and ``b`` to if given.

    .. seealso:: :func:`numpy.testing.assert_array_max_ulp`
    """
    numpy.testing.assert_array_max_ulp(cupy.asnumpy(a), cupy.asnumpy(b), maxulp=maxulp, dtype=dtype)