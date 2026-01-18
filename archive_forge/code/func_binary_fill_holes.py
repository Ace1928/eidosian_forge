import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def binary_fill_holes(input, structure=None, output=None, origin=0):
    """Fill the holes in binary objects.

    Args:
        input (cupy.ndarray): N-D binary array with holes to be filled.
        structure (cupy.ndarray, optional):  Structuring element used in the
            computation; large-size elements make computations faster but may
            miss holes separated from the background by thin regions. The
            default element (with a square connectivity equal to one) yields
            the intuitive result where all holes in the input have been filled.
        output (cupy.ndarray, dtype or None, optional): Array of the same shape
            as input, into which the output is placed. By default, a new array
            is created.
        origin (int, tuple of ints, optional): Position of the structuring
            element.

    Returns:
        cupy.ndarray: Transformation of the initial image ``input`` where holes
        have been filled.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_fill_holes`
    """
    mask = cupy.logical_not(input)
    tmp = cupy.zeros(mask.shape, bool)
    inplace = isinstance(output, cupy.ndarray)
    if inplace:
        binary_dilation(tmp, structure, -1, mask, output, 1, origin, brute_force=True)
        cupy.logical_not(output, output)
    else:
        output = binary_dilation(tmp, structure, -1, mask, None, 1, origin, brute_force=True)
        cupy.logical_not(output, output)
        return output