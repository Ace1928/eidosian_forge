import numpy
import cupy
def array2string(a, *args, **kwargs):
    """Return a string representation of an array.


    .. seealso:: :func:`numpy.array2string`

    """
    return numpy.array2string(cupy.asnumpy(a), *args, **kwargs)