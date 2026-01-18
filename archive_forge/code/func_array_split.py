import numpy
from cupy import _core
def array_split(ary, indices_or_sections, axis=0):
    """Splits an array into multiple sub arrays along a given axis.

    This function is almost equivalent to :func:`cupy.split`. The only
    difference is that this function allows an integer sections that does not
    evenly divide the axis.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.array_split`

    """
    return _core.array_split(ary, indices_or_sections, axis)