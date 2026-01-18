import cupy
def const_ext(x, n, axis=-1):
    """
    Constant extension at the boundaries of an array

    Generate a new ndarray that is a constant extension of `x` along an axis.
    The extension repeats the values at the first and last element of
    the axis.

    Parameters
    ----------
    x : ndarray
        The array to be extended.
    n : int
        The number of elements by which to extend `x` at each end of the axis.
    axis : int, optional
        The axis along which to extend `x`. Default is -1.

    Examples
    --------
    >>> from cupyx.scipy.signal._arraytools import const_ext
    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])
    >>> const_ext(a, 2)
    array([[ 1,  1,  1,  2,  3,  4,  5,  5,  5],
           [ 0,  0,  0,  1,  4,  9, 16, 16, 16]])
    """
    if n < 1:
        return x
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    ones_shape = [1] * x.ndim
    ones_shape[axis] = n
    ones = cupy.ones(ones_shape, dtype=x.dtype)
    left_ext = ones * left_end
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = ones * right_end
    ext = cupy.concatenate((left_ext, x, right_ext), axis=axis)
    return ext