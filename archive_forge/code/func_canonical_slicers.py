import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def canonical_slicers(sliceobj, shape, check_inds=True):
    """Return canonical version of `sliceobj` for array shape `shape`

    `sliceobj` is a slicer for an array ``A`` implied by `shape`.

    * Expand `sliceobj` with ``slice(None)`` to add any missing (implied) axes
      in `sliceobj`
    * Find any slicers in `sliceobj` that do a full axis slice and replace by
      ``slice(None)``
    * Replace any floating point values for slicing with integers
    * Replace negative integer slice values with equivalent positive integers.

    Does not handle fancy indexing (indexing with arrays or array-like indices)

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    shape : sequence
        shape of array that will be indexed by `sliceobj`
    check_inds : {True, False}, optional
        Whether to check if integer indices are out of bounds

    Returns
    -------
    can_slicers : tuple
        version of `sliceobj` for which Ellipses have been expanded, missing
        (implied) dimensions have been appended, and slice objects equivalent
        to ``slice(None)`` have been replaced by ``slice(None)``, integer axes
        have been checked, and negative indices set to positive equivalent
    """
    if not isinstance(sliceobj, tuple):
        sliceobj = (sliceobj,)
    if is_fancy(sliceobj):
        raise ValueError('Cannot handle fancy indexing')
    can_slicers = []
    n_dim = len(shape)
    n_real = 0
    for i, slicer in enumerate(sliceobj):
        if slicer is None:
            can_slicers.append(None)
            continue
        if slicer == Ellipsis:
            remaining = sliceobj[i + 1:]
            if Ellipsis in remaining:
                raise ValueError('More than one Ellipsis in slicing expression')
            real_remaining = [r for r in remaining if r is not None]
            n_ellided = n_dim - n_real - len(real_remaining)
            can_slicers.extend((slice(None),) * n_ellided)
            n_real += n_ellided
            continue
        dim_len = shape[n_real]
        n_real += 1
        try:
            slicer = int(slicer)
        except TypeError:
            if slicer != slice(None):
                if slicer.stop == dim_len and slicer.start in (None, 0) and (slicer.step in (None, 1)):
                    slicer = slice(None)
        else:
            if slicer < 0:
                slicer = dim_len + slicer
            elif check_inds and slicer >= dim_len:
                raise ValueError('Integer index %d to large' % slicer)
        can_slicers.append(slicer)
    if n_real < n_dim:
        can_slicers.extend((slice(None),) * (n_dim - n_real))
    return tuple(can_slicers)