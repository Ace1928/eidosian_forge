import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def medfilt2d(input, kernel_size=3):
    """Median filter a 2-dimensional array.

    Apply a median filter to the `input` array using a local window-size given
    by `kernel_size` (must be odd). The array is zero-padded automatically.

    Args:
        input (cupy.ndarray): A 2-dimensional input array.
        kernel_size (int of list of ints of length 2): Gives the size of the
            median filter window in each dimension. Elements of `kernel_size`
            should be odd. If `kernel_size` is a scalar, then this scalar is
            used as the size in each dimension. Default is a kernel of size
            (3, 3).

    Returns:
        cupy.ndarray: An array the same size as input containing the median
        filtered result.

    See also
    --------
    .. seealso:: :func:`cupyx.scipy.ndimage.median_filter`
    .. seealso:: :func:`cupyx.scipy.signal.medfilt`
    .. seealso:: :func:`scipy.signal.medfilt2d`
    """
    if input.dtype.kind == 'c':
        raise ValueError('complex types not supported')
    if input.dtype.char == 'e':
        raise ValueError('float16 type not supported')
    if input.dtype.kind == 'b':
        raise ValueError('bool type not supported')
    if input.ndim != 2:
        raise ValueError('input must be 2d')
    kernel_size = _get_kernel_size(kernel_size, input.ndim)
    order = kernel_size[0] * kernel_size[1] // 2
    return _filters.rank_filter(input, order, size=kernel_size, mode='constant')