import numpy
from cupy import _core
def dsplit(ary, indices_or_sections):
    """Splits an array into multiple sub arrays along the third axis.

    This is equivalent to ``split`` with ``axis=2``.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.dsplit`

    """
    if ary.ndim <= 2:
        raise ValueError('Cannot dsplit an array with less than 3 dimensions')
    return split(ary, indices_or_sections, 2)