import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    """One-dimensional Gaussian filter along the given axis.

    The lines of the array along the given axis are filtered with a Gaussian
    filter of the given standard deviation.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar): Standard deviation for Gaussian kernel.
        axis (int): The axis of input along which to calculate. Default is -1.
        order (int): An order of ``0``, the default, corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_filter1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    radius = int(float(truncate) * float(sigma) + 0.5)
    weights_dtype = _util._init_weights_dtype(input)
    weights = _gaussian_kernel1d(sigma, int(order), radius, dtype=weights_dtype)
    return correlate1d(input, weights, axis, output, mode, cval)