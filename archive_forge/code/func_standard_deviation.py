import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def standard_deviation(input, labels=None, index=None):
    """Calculates the standard deviation of the values of an n-D image array,
    optionally at specified sub-regions.

    Args:
        input (cupy.ndarray): Nd-image data to process.
        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.
            If not None, must be same shape as `input`.
        index (cupy.ndarray or None): `labels` to include in output. If None
            (default), all values where `labels` is non-zero are used.

    Returns:
        standard_deviation (cupy.ndarray): standard deviation of values, for
        each sub-region if `labels` and `index` are specified.

    .. seealso:: :func:`scipy.ndimage.standard_deviation`
    """
    return cupy.sqrt(variance(input, labels, index))