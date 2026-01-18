import cupy
from cupy import _core
def column_stack(tup):
    """Stacks 1-D and 2-D arrays as columns into a 2-D array.

    A 1-D array is first converted to a 2-D column array. Then, the 2-D arrays
    are concatenated along the second axis.

    Args:
        tup (sequence of arrays): 1-D or 2-D arrays to be stacked.

    Returns:
        cupy.ndarray: A new 2-D array of stacked columns.

    .. seealso:: :func:`numpy.column_stack`

    """
    if any((not isinstance(a, cupy.ndarray) for a in tup)):
        raise TypeError('Only cupy arrays can be column stacked')
    lst = list(tup)
    for i, a in enumerate(lst):
        if a.ndim == 1:
            a = a[:, cupy.newaxis]
            lst[i] = a
        elif a.ndim != 2:
            raise ValueError('Only 1 or 2 dimensional arrays can be column stacked')
    return concatenate(lst, axis=1)