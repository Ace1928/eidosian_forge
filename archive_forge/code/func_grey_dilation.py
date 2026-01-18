import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def grey_dilation(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    """Calculates a greyscale dilation.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the greyscale dilation. Optional if ``footprint`` or
            ``structure`` is provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for greyscale dilation. Non-zero values
            give the set of neighbors of the center over which maximum is
            chosen.
        structure (array of ints): Structuring element used for the greyscale
            dilation. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of greyscale dilation.

    .. seealso:: :func:`scipy.ndimage.grey_dilation`
    """
    if size is None and footprint is None and (structure is None):
        raise ValueError('size, footprint or structure must be specified')
    if structure is not None:
        structure = cupy.array(structure)
        structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    if footprint is not None:
        footprint = cupy.array(footprint)
        footprint = footprint[tuple([slice(None, None, -1)] * footprint.ndim)]
    origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    for i in range(len(origin)):
        origin[i] = -origin[i]
        if footprint is not None:
            sz = footprint.shape[i]
        elif structure is not None:
            sz = structure.shape[i]
        elif numpy.isscalar(size):
            sz = size
        else:
            sz = size[i]
        if sz % 2 == 0:
            origin[i] -= 1
    return _filters._min_or_max_filter(input, size, footprint, structure, output, mode, cval, origin, 'max')