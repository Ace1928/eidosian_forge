from ._internal import NDArrayBase
from ..base import _Null
def interleaved_matmul_selfatt_valatt(queries_keys_values=None, attention=None, heads=_Null, out=None, name=None, **kwargs):
    """Compute the matrix multiplication between the projections of
    values and the attention weights in multihead attention use as self attention.

    the inputs must be a tensor of interleaved projections
    of queries, keys and values following the layout:
    (seq_length, batch_size, num_heads * head_dim * 3)

    and the attention weights following the layout:
    (batch_size, seq_length, seq_length)

    the equivalent code would be:
    tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
    v_proj = mx.nd.transpose(tmp[:,:,:,2,:], axes=(1, 2, 0, 3))
    v_proj = mx.nd.reshape(v_proj, shape=(-1, 0, 0), reverse=True)
    output = mx.nd.batch_dot(attention, v_proj, transpose_b=True)
    output = mx.nd.reshape(output, shape=(-1, num_heads, 0, 0), reverse=True)
    output = mx.nd.transpose(output, axes=(0, 2, 1, 3))
    output = mx.nd.reshape(output, shape=(0, 0, -1))


    Defined in ../src/operator/contrib/transformer.cc:L709

    Parameters
    ----------
    queries_keys_values : NDArray
        Queries, keys and values interleaved
    attention : NDArray
        Attention maps
    heads : int, required
        Set number of heads

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)