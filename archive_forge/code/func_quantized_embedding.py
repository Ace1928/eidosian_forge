from ._internal import NDArrayBase
from ..base import _Null
def quantized_embedding(data=None, weight=None, min_weight=None, max_weight=None, input_dim=_Null, output_dim=_Null, dtype=_Null, sparse_grad=_Null, out=None, name=None, **kwargs):
    """Maps integer indices to int8 vector representations (embeddings).


    Defined in ../src/operator/quantization/quantized_indexing_op.cc:L133

    Parameters
    ----------
    data : NDArray
        The input array to the embedding operator.
    weight : NDArray
        The embedding weight matrix.
    min_weight : NDArray
        Minimum value of data.
    max_weight : NDArray
        Maximum value of data.
    input_dim : int, required
        Vocabulary size of the input indices.
    output_dim : int, required
        Dimension of the embedding vectors.
    dtype : {'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},optional, default='float32'
        Data type of weight.
    sparse_grad : boolean, optional, default=0
        Compute row sparse gradient in the backward calculation. If set to True, the grad's storage type is row_sparse.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)