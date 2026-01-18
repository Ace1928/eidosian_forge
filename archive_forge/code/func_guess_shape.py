import numpy as np
from .base import product
from .. import h5s, h5r, _selector
def guess_shape(sid):
    """ Given a dataspace, try to deduce the shape of the selection.

    Returns one of:
        * A tuple with the selection shape, same length as the dataspace
        * A 1D selection shape for point-based and multiple-hyperslab selections
        * None, for unselected scalars and for NULL dataspaces
    """
    sel_class = sid.get_simple_extent_type()
    sel_type = sid.get_select_type()
    if sel_class == h5s.NULL:
        return None
    elif sel_class == h5s.SCALAR:
        if sel_type == h5s.SEL_NONE:
            return None
        if sel_type == h5s.SEL_ALL:
            return tuple()
    elif sel_class != h5s.SIMPLE:
        raise TypeError('Unrecognized dataspace class %s' % sel_class)
    N = sid.get_select_npoints()
    rank = len(sid.shape)
    if sel_type == h5s.SEL_NONE:
        return (0,) * rank
    elif sel_type == h5s.SEL_ALL:
        return sid.shape
    elif sel_type == h5s.SEL_POINTS:
        return (N,)
    elif sel_type != h5s.SEL_HYPERSLABS:
        raise TypeError('Unrecognized selection method %s' % sel_type)
    if N == 0:
        return (0,) * rank
    bottomcorner, topcorner = (np.array(x) for x in sid.get_select_bounds())
    boxshape = topcorner - bottomcorner + np.ones((rank,))

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
    shape = tuple((get_n_axis(sid, x) for x in range(rank)))
    if product(shape) != N:
        return (N,)
    return shape