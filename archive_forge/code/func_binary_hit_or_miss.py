import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def binary_hit_or_miss(input, structure1=None, structure2=None, output=None, origin1=0, origin2=None):
    """
    Multidimensional binary hit-or-miss transform.

    The hit-or-miss transform finds the locations of a given pattern
    inside the input image.

    Args:
        input (cupy.ndarray): Binary image where a pattern is to be detected.
        structure1 (cupy.ndarray, optional): Part of the structuring element to
            be fitted to the foreground (non-zero elements) of ``input``. If no
            value is provided, a structure of square connectivity 1 is chosen.
        structure2 (cupy.ndarray, optional): Second part of the structuring
            element that has to miss completely the foreground. If no value is
            provided, the complementary of ``structure1`` is taken.
        output (cupy.ndarray, dtype or None, optional): Array of the same shape
            as input, into which the output is placed. By default, a new array
            is created.
        origin1 (int or tuple of ints, optional): Placement of the first part
            of the structuring element ``structure1``, by default 0 for a
            centered structure.
        origin2 (int or tuple of ints or None, optional): Placement of the
            second part of the structuring element ``structure2``, by default 0
            for a centered structure. If a value is provided for ``origin1``
            and not for ``origin2``, then ``origin2`` is set to ``origin1``.

    Returns:
        cupy.ndarray: Hit-or-miss transform of ``input`` with the given
        structuring element (``structure1``, ``structure2``).

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.binary_hit_or_miss`
    """
    if structure1 is None:
        structure1 = generate_binary_structure(input.ndim, 1)
    if structure2 is None:
        structure2 = cupy.logical_not(structure1)
    origin1 = _util._fix_sequence_arg(origin1, input.ndim, 'origin1', int)
    if origin2 is None:
        origin2 = origin1
    else:
        origin2 = _util._fix_sequence_arg(origin2, input.ndim, 'origin2', int)
    tmp1 = _binary_erosion(input, structure1, 1, None, None, 0, origin1, 0, False)
    inplace = isinstance(output, cupy.ndarray)
    result = _binary_erosion(input, structure2, 1, None, output, 0, origin2, 1, False)
    if inplace:
        cupy.logical_not(output, output)
        cupy.logical_and(tmp1, output, output)
    else:
        cupy.logical_not(result, result)
        return cupy.logical_and(tmp1, result)