from ._internal import NDArrayBase
from ..base import _Null
def interleaved_matmul_selfatt_qk(queries_keys_values=None, heads=_Null, out=None, name=None, **kwargs):
    """Compute the matrix multiplication between the projections of
    queries and keys in multihead attention use as self attention.

    the input must be a single tensor of interleaved projections
    of queries, keys and values following the layout:
    (seq_length, batch_size, num_heads * head_dim * 3)

    the equivalent code would be:
    tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
    q_proj = mx.nd.transpose(tmp[:,:,:,0,:], axes=(1, 2, 0, 3))
    q_proj = mx.nd.reshape(q_proj, shape=(-1, 0, 0), reverse=True)
    q_proj = mx.nd.contrib.div_sqrt_dim(q_proj)
    k_proj = mx.nd.transpose(tmp[:,:,:,1,:], axes=(1, 2, 0, 3))
    k_proj = mx.nd.reshap(k_proj, shape=(-1, 0, 0), reverse=True)
    output = mx.nd.batch_dot(q_proj, k_proj, transpose_b=True)


    Defined in ../src/operator/contrib/transformer.cc:L665

    Parameters
    ----------
    queries_keys_values : NDArray
        Interleaved queries, keys and values
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