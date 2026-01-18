from ._internal import NDArrayBase
from ..base import _Null
def elemwise_mul(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Multiplies arguments element-wise.

    The storage type of ``elemwise_mul`` output depends on storage types of inputs

       - elemwise_mul(default, default) = default
       - elemwise_mul(row_sparse, row_sparse) = row_sparse
       - elemwise_mul(default, row_sparse) = row_sparse
       - elemwise_mul(row_sparse, default) = row_sparse
       - elemwise_mul(csr, csr) = csr
       - otherwise, ``elemwise_mul`` generates output with default storage



    Parameters
    ----------
    lhs : NDArray
        first input
    rhs : NDArray
        second input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)