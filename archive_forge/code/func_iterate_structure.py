import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
def iterate_structure(structure, iterations, origin=None):
    """Iterate a structure by dilating it with itself.

    Args:
        structure(array_like): Structuring element (an array of bools,
            for example), to be dilated with itself.
        iterations(int): The number of dilations performed on the structure
            with itself.
        origin(int or tuple of int, optional): If origin is None, only the
            iterated structure is returned. If not, a tuple of the iterated
            structure and the modified origin is returned.

    Returns:
        cupy.ndarray: A new structuring element obtained by dilating
        ``structure`` (``iterations`` - 1) times with itself.

    .. seealso:: :func:`scipy.ndimage.iterate_structure`
    """
    if iterations < 2:
        return structure.copy()
    ni = iterations - 1
    shape = [ii + ni * (ii - 1) for ii in structure.shape]
    pos = [ni * (structure.shape[ii] // 2) for ii in range(len(shape))]
    slc = tuple((slice(pos[ii], pos[ii] + structure.shape[ii], None) for ii in range(len(shape))))
    out = cupy.zeros(shape, bool)
    out[slc] = structure != 0
    out = binary_dilation(out, structure, iterations=ni)
    if origin is None:
        return out
    else:
        origin = _util._fix_sequence_arg(origin, structure.ndim, 'origin', int)
        origin = [iterations * o for o in origin]
        return (out, origin)