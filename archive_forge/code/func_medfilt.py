import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def medfilt(volume, kernel_size=None):
    """Perform a median filter on an N-dimensional array.

    Apply a median filter to the input array using a local window-size
    given by `kernel_size`. The array will automatically be zero-padded.

    Args:
        volume (cupy.ndarray): An N-dimensional input array.
        kernel_size (int or list of ints): Gives the size of the median filter
            window in each dimension. Elements of `kernel_size` should be odd.
            If `kernel_size` is a scalar, then this scalar is used as the size
            in each dimension. Default size is 3 for each dimension.

    Returns:
        cupy.ndarray: An array the same size as input containing the median
        filtered result.

    .. seealso:: :func:`cupyx.scipy.ndimage.median_filter`
    .. seealso:: :func:`scipy.signal.medfilt`
    """
    if volume.dtype.kind == 'c':
        raise ValueError('complex types not supported')
    if volume.dtype.char == 'e':
        raise ValueError('float16 type not supported')
    if volume.dtype.kind == 'b':
        raise ValueError('bool type not supported')
    kernel_size = _get_kernel_size(kernel_size, volume.ndim)
    if any((k > s for k, s in zip(kernel_size, volume.shape))):
        warnings.warn('kernel_size exceeds volume extent: volume will be zero-padded')
    size = internal.prod(kernel_size)
    return _filters.rank_filter(volume, size // 2, size=kernel_size, mode='constant')