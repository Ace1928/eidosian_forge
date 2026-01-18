import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs):
    """Multi-dimensional Laplace filter using Gaussian second derivatives.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """

    def derivative2(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval, **kwargs)
    return generic_laplace(input, derivative2, output, mode, cval)