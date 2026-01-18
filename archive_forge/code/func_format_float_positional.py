import numpy
import cupy
def format_float_positional(x, *args, **kwargs):
    """Format a floating-point scalar as a decimal string in positional notation.

    See :func:`numpy.format_float_positional` for the list of arguments.

    .. seealso:: :func:`numpy.format_float_positional`

    """
    return numpy.format_float_positional(cupy.asnumpy(x), *args, **kwargs)