import functools
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
from keras.src.utils import jax_utils
def axis_shape_dims_for_broadcast_in_dim(axis, input_shape, insert_dims):
    """Turn the `axis` argument to the arguments needed by `broadcast_in_dim`.

    Args:
        axis: single int or a tuple of ints for the axis argument. The list of
          dimensions to reduce or insert.
        input_shape: the shape of the input as a tuple ints.
        insert_dims: `False` turns dimensions in `axis` to 1s (use case:
          reduction along `axis` with `keep_dims=True`). `True`, inserts 1s
          according to `axis` (use case: `expand_dims`).
    Returns:
        A tuple of three lists
        - The canonical value for `axis`: always a list, negative values have
          been resolved and values are sorted in ascending order.
        - The output shape: `input_shape` with 1s at the indices in `axis`, for
          use as the `shape` argument of `broadcast_in_dim`.
        - The broadcast dimensions: list of dimensions not in `axis`, for use as
          the `broadcast_dimensions` argument of `broadcast_in_dim`.
    """
    if axis is None:
        raise ValueError('Received `None` value for `axis`')
    if isinstance(axis, int):
        axis = (axis,)
    if len(set(axis)) != len(axis):
        raise ValueError(f'Repeated axis in `axis`: {axis}')
    result_dims = len(input_shape)
    if insert_dims:
        result_dims += len(axis)
    canonical_axis = []
    for a in axis:
        if not -result_dims <= a < result_dims:
            raise ValueError(f'In `axis`, axis {a} is out of bounds for array of dimension {result_dims}')
        if a < 0:
            a = a + result_dims
        canonical_axis.append(a)
    if len(set(canonical_axis)) != len(canonical_axis):
        raise ValueError(f'Repeated axis in `axis`: {canonical_axis}')
    canonical_axis = sorted(canonical_axis)
    output_shape = list(input_shape)
    for i in canonical_axis:
        if insert_dims:
            output_shape.insert(i, 1)
        else:
            output_shape[i] = 1
    broadcast_dims = [i for i in range(result_dims) if i not in canonical_axis]
    return (canonical_axis, output_shape, broadcast_dims)