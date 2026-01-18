import numpy
import cupy
def format_float_scientific(x, *args, **kwargs):
    """Format a floating-point scalar as a decimal string in scientific notation.

    See :func:`numpy.format_float_scientific` for the list of arguments.

    .. seealso:: :func:`numpy.format_float_scientific`

    """
    return numpy.format_float_scientific(cupy.asnumpy(x), *args, **kwargs)