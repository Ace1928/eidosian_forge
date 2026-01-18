import cupy
def axis_assign(a, b, start=None, stop=None, step=None, axis=-1):
    """Take a slice along axis 'axis' from 'a' and set it to 'b' in-place.

    Parameters
    ----------
    a : numpy.ndarray
        The array to be sliced.
    b : cupy.ndarray
        The array to be assigned.
    start, stop, step : int or None
        The slice parameters.
    axis : int, optional
        The axis of `a` to be sliced.

    Examples
    --------
    >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b1 = array([[-1], [-4], [-7]])
    >>> axis_assign(a, b1, start=0, stop=1, axis=1)
    array([[-1, 2, 3],
           [-4, 5, 6],
           [-7, 8, 9]])

    Notes
    -----
    The keyword arguments start, stop and step are used by calling
    slice(start, stop, step). This implies axis_assign() does not
    handle its arguments the exactly the same as indexing. To assign
    a single index k, for example, use
        axis_assign(a, start=k, stop=k+1)
    In this case, the length of the axis 'axis' in the result will
    be 1; the trivial dimension is not removed. (Use numpy.squeeze()
    to remove trivial axes.)

    This function works in-place and will modify the values contained in `a`
    """
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    a[tuple(a_slice)] = b
    return a