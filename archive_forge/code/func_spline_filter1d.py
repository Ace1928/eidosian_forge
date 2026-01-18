import math
import warnings
import cupy
import numpy
from cupy import _core
from cupy._core import internal
from cupy.cuda import runtime
from cupyx import _texture
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _interp_kernels
from cupyx.scipy.ndimage import _spline_prefilter_core
def spline_filter1d(input, order=3, axis=-1, output=cupy.float64, mode='mirror'):
    """
    Calculate a 1-D spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    Args:
        input (cupy.ndarray): The input array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        axis (int): The axis along which the spline filter is applied. Default
            is the last axis.
        output (cupy.ndarray or dtype, optional): The array in which to place
            the output, or the dtype of the returned array. Default is
            ``numpy.float64``.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).

    Returns:
        cupy.ndarray: The result of prefiltering the input.

    .. seealso:: :func:`scipy.spline_filter1d`
    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    x = input
    ndim = x.ndim
    axis = internal._normalize_axis_index(axis, ndim)
    run_kernel = not (order < 2 or x.ndim == 0 or x.shape[axis] == 1)
    if not run_kernel:
        output = _util._get_output(output, input)
        _core.elementwise_copy(x, output)
        return output
    temp, data_dtype, output_dtype = _get_spline_output(x, output)
    data_type = cupy._core._scalar.get_typename(temp.dtype)
    pole_type = cupy._core._scalar.get_typename(temp.real.dtype)
    index_type = _util._get_inttype(input)
    index_dtype = cupy.int32 if index_type == 'int' else cupy.int64
    n_samples = x.shape[axis]
    n_signals = x.size // n_samples
    info = cupy.array((n_signals, n_samples) + x.shape, dtype=index_dtype)
    block_size = max(2 ** math.ceil(numpy.log2(n_samples / 32)), 8)
    kern = _spline_prefilter_core.get_raw_spline1d_kernel(axis, ndim, mode, order=order, index_type=index_type, data_type=data_type, pole_type=pole_type, block_size=block_size)
    block = (block_size,)
    grid = ((n_signals + block[0] - 1) // block[0],)
    poles = _spline_prefilter_core.get_poles(order=order)
    temp *= _spline_prefilter_core.get_gain(poles)
    kern(grid, block, (temp, info))
    if isinstance(output, cupy.ndarray) and temp is not output:
        _core.elementwise_copy(temp, output)
        return output
    return temp.astype(output_dtype, copy=False)