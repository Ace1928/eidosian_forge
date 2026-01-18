import numpy as np
from .base import product
from .. import h5s, h5r, _selector
def get_n_axis(sid, axis):
    """ Determine the number of elements selected along a particular axis.

        To do this, we "mask off" the axis by making a hyperslab selection
        which leaves only the first point along the axis.  For a 2D dataset
        with selection box shape (X, Y), for axis 1, this would leave a
        selection of shape (X, 1).  We count the number of points N_leftover
        remaining in the selection and compute the axis selection length by
        N_axis = N/N_leftover.
        """
    if boxshape[axis] == 1:
        return 1
    start = bottomcorner.copy()
    start[axis] += 1
    count = boxshape.copy()
    count[axis] -= 1
    masked_sid = sid.copy()
    masked_sid.select_hyperslab(tuple(start), tuple(count), op=h5s.SELECT_NOTB)
    N_leftover = masked_sid.get_select_npoints()
    return N // N_leftover