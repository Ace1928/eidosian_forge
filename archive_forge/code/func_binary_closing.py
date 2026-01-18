import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def binary_closing(input, structure=None, iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False):
    """
    Multidimensional binary closing with the given structuring element.

    The *closing* of an input image by a structuring element is the
    *erosion* of the *dilation* of the image by the structuring element.

    Args:
        input(cupy.ndarray): The input binary array to be closed.
            Non-zero (True) elements form the subset to be closed.
        structure(cupy.ndarray, optional): The structuring element used for the
            closing. Non-zero elements are considered True. If no structuring
            element is provided an element is generated with a square
            connectivity equal to one. (Default value = None).
        iterations(int, optional): The closing is repeated ``iterations`` times
            (one, by default). If iterations is less than 1, the closing is
            repeated until the result does not change anymore. Only an integer
            of iterations is accepted.
        output(cupy.ndarray, optional): Array of the same shape as input, into
            which the output is placed. By default, a new array is created.
        origin(int or tuple of ints, optional): Placement of the filter, by
            default 0.
        mask(cupy.ndarray or None, optional): If a mask is given, only those
            elements with a True value at the corresponding mask element are
            modified at each iteration. (Default value = None)
        border_value(int (cast to 0 or 1), optional): Value at the
            border in the output array. (Default value = 0)
        brute_force(boolean, optional): Memory condition: if False, only the
            pixels whose value was changed in the last iteration are tracked as
            candidates to be updated (dilated) in the current iteration; if
            True all pixels are considered as candidates for closing,
            regardless of what happened in the previous iteration.

    Returns:
        cupy.ndarray: The result of binary closing.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_closing`
    """
    if structure is None:
        rank = input.ndim
        structure = generate_binary_structure(rank, 1)
    tmp = binary_dilation(input, structure, iterations, mask, None, border_value, origin, brute_force)
    return binary_erosion(tmp, structure, iterations, mask, output, border_value, origin, brute_force)