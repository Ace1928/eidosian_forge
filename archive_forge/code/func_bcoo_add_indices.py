import functools
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
from keras.src.utils import jax_utils
def bcoo_add_indices(x1, x2, sum_duplicates):
    """Add the indices of `x2` to `x1` with zero values.

    Args:
        x1: `BCOO` tensor to add indices to.
        x2: `BCOO` tensor to take the indices to add to x1.
        sum_duplicates: if `True` calls `bcoo_sum_duplicates` on the output.
    Returns:
        a `BCOO` tensor equal to `x1` but with extra zeros at indices in `x2`
        that were missing in `x1`.
    """
    x2_zeros = jnp.zeros(x2.data.shape, x1.data.dtype)
    concat_axis = len(x1.indices.shape) - 2
    output_indices = jnp.concatenate([x1.indices, x2.indices], axis=concat_axis)
    output_data = jnp.concatenate([x1.data, x2_zeros], axis=concat_axis)
    output = jax_sparse.BCOO((output_data, output_indices), shape=x1.shape)
    if sum_duplicates:
        output = jax_sparse.bcoo_sum_duplicates(output)
    return output