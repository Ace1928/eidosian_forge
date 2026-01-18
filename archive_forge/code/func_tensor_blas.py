import numpy as np
from . import helpers
def tensor_blas(view_left, input_left, view_right, input_right, index_result, idx_removed):
    """
    Computes the dot product between two tensors, attempts to use np.dot and
    then tensordot if that fails.

    Parameters
    ----------
    view_left : array_like
        The left hand view
    input_left : str
        Indices of the left view
    view_right : array_like
        The right hand view
    input_right : str
        Indices of the right view
    index_result : str
        The resulting indices
    idx_removed : set
        Indices removed in the contraction

    Returns
    -------
    type : array
        The resulting BLAS operation.

    Notes
    -----
    Interior function for tensor BLAS.

    This function will attempt to use `np.dot` by the iterating through the
    four possible transpose cases. If this fails all inner and matrix-vector
    operations will be handed off to einsum while all matrix-matrix operations will
    first copy the data, perform the DGEMM, and then copy the data to the required
    order.

    Examples
    --------

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4)
    >>> tmp = tensor_blas(a, 'ij', b, 'jk', 'ik', set('j'))
    >>> np.allclose(tmp, np.dot(a, b))

    """
    idx_removed = set(idx_removed)
    keep_left = set(input_left) - idx_removed
    keep_right = set(input_right) - idx_removed
    dimension_dict = {}
    for i, s in zip(input_left, view_left.shape):
        dimension_dict[i] = s
    for i, s in zip(input_right, view_right.shape):
        dimension_dict[i] = s
    rs = len(idx_removed)
    dim_left = helpers.compute_size_by_dict(keep_left, dimension_dict)
    dim_right = helpers.compute_size_by_dict(keep_right, dimension_dict)
    dim_removed = helpers.compute_size_by_dict(idx_removed, dimension_dict)
    tensor_result = input_left + input_right
    for s in idx_removed:
        tensor_result = tensor_result.replace(s, '')
    if input_left == input_right:
        new_view = np.dot(view_left.ravel(), view_right.ravel())
    elif input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(view_left.reshape(dim_left, dim_removed), view_right.reshape(dim_removed, dim_right))
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T, view_right.reshape(dim_right, dim_removed).T)
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(view_left.reshape(dim_left, dim_removed), view_right.reshape(dim_right, dim_removed).T)
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(view_left.reshape(dim_removed, dim_left).T, view_right.reshape(dim_removed, dim_right))
    else:
        left_pos, right_pos = ((), ())
        for s in idx_removed:
            left_pos += (input_left.find(s),)
            right_pos += (input_right.find(s),)
        new_view = np.tensordot(view_left, view_right, axes=(left_pos, right_pos))
    tensor_shape = tuple((dimension_dict[x] for x in tensor_result))
    if new_view.shape != tensor_shape:
        if len(tensor_result) > 0:
            new_view.shape = tensor_shape
        else:
            new_view = np.squeeze(new_view)
    if tensor_result != index_result:
        new_view = np.einsum(tensor_result + '->' + index_result, new_view)
    return new_view