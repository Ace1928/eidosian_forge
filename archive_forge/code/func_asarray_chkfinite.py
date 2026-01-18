import cupy
from cupy import _core
def asarray_chkfinite(a, dtype=None, order=None):
    """Converts the given input to an array,
    and raises an error if the input contains NaNs or Infs.

    Args:
        a: array like.
        dtype: data type, optional
        order: {'C', 'F', 'A', 'K'}, optional

    Returns:
        cupy.ndarray: An array on the current device.

    .. note::
        This function performs device synchronization.

    .. seealso:: :func:`numpy.asarray_chkfinite`

    """
    a = cupy.asarray(a, dtype=dtype, order=order)
    if not cupy.isfinite(a).all():
        raise ValueError('array must not contain Infs or NaNs')
    return a