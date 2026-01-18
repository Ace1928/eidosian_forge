import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def predict_shape(sliceobj, in_shape):
    """Predict shape of array from slicing array shape `shape` with `sliceobj`

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    in_shape : sequence
        shape of array that could be sliced by `sliceobj`

    Returns
    -------
    out_shape : tuple
        predicted shape arising from slicing array shape `in_shape` with
        `sliceobj`
    """
    if not isinstance(sliceobj, tuple):
        sliceobj = (sliceobj,)
    sliceobj = canonical_slicers(sliceobj, in_shape)
    out_shape = []
    real_no = 0
    for slicer in sliceobj:
        if slicer is None:
            out_shape.append(1)
            continue
        real_no += 1
        try:
            slicer = int(slicer)
        except TypeError:
            out_shape.append(slice2len(slicer, in_shape[real_no - 1]))
    return tuple(out_shape)