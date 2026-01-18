import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def binary_erosion(input, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False):
    """Multidimensional binary erosion with a given structuring element.

    Binary erosion is a mathematical morphology operation used for image
    processing.

    Args:
        input(cupy.ndarray): The input binary array_like to be eroded.
            Non-zero (True) elements form the subset to be eroded.
        structure(cupy.ndarray, optional): The structuring element used for the
            erosion. Non-zero elements are considered True. If no structuring
            element is provided an element is generated with a square
            connectivity equal to one. (Default value = None).
        iterations(int, optional): The erosion is repeated ``iterations`` times
            (one, by default). If iterations is less than 1, the erosion is
            repeated until the result does not change anymore. Only an integer
            of iterations is accepted.
        mask(cupy.ndarray or None, optional): If a mask is given, only those
            elements with a True value at the corresponding mask element are
            modified at each iteration. (Default value = None)
        output(cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        border_value(int (cast to 0 or 1), optional): Value at the
            border in the output array. (Default value = 0)
        origin(int or tuple of ints, optional): Placement of the filter, by
            default 0.
        brute_force(boolean, optional): Memory condition: if False, only the
            pixels whose value was changed in the last iteration are tracked as
            candidates to be updated (eroded) in the current iteration; if
            True all pixels are considered as candidates for erosion,
            regardless of what happened in the previous iteration.

    Returns:
        cupy.ndarray: The result of binary erosion.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_erosion`
    """
    return _binary_erosion(input, structure, iterations, mask, output, border_value, origin, 0, brute_force)