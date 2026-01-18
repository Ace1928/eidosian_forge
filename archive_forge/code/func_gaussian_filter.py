import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    """Multi-dimensional Gaussian filter.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        order (int or sequence of scalar): An order of ``0``, the default,
            corresponds to convolution with a Gaussian kernel. A positive order
            corresponds to convolution with that derivative of a Gaussian. A
            single value applies to all axes.
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

    .. seealso:: :func:`scipy.ndimage.gaussian_filter`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    sigmas = _util._fix_sequence_arg(sigma, input.ndim, 'sigma', float)
    orders = _util._fix_sequence_arg(order, input.ndim, 'order', int)
    truncate = float(truncate)
    weights_dtype = _util._init_weights_dtype(input)

    def get(param):
        sigma, order = param
        radius = int(truncate * float(sigma) + 0.5)
        if radius <= 0:
            return None
        return _gaussian_kernel1d(sigma, order, radius, dtype=weights_dtype)
    return _run_1d_correlates(input, list(zip(sigmas, orders)), get, output, mode, cval, 0)