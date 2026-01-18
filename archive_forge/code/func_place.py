import numpy
import cupy
from cupy import _core
def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values.

    This function uses the first N elements of `vals`, where N is the number
    of true values in `mask`.

    Args:
        arr (cupy.ndarray): Array to put data into.
        mask (array-like): Boolean mask array. Must have the same size as `a`.
        vals (array-like): Values to put into `a`. Only the first
            N elements are used, where N is the number of True values in
            `mask`. If `vals` is smaller than N, it will be repeated, and if
            elements of `a` are to be masked, this sequence must be non-empty.

    Examples
    --------
    >>> arr = np.arange(6).reshape(2, 3)
    >>> np.place(arr, arr>2, [44, 55])
    >>> arr
    array([[ 0,  1,  2],
           [44, 55, 44]])

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.place`
    """
    mask = cupy.asarray(mask)
    if arr.size != mask.size:
        raise ValueError('Mask and data must be the same size.')
    vals = cupy.asarray(vals)
    mask_indices = mask.ravel().nonzero()[0]
    if mask_indices.size == 0:
        return
    if vals.size == 0:
        raise ValueError('Cannot insert from an empty array.')
    arr.put(mask_indices, vals, mode='wrap')