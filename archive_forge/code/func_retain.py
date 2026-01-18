from ._internal import NDArrayBase
from ..base import _Null
def retain(data=None, indices=None, out=None, name=None, **kwargs):
    """Pick rows specified by user input index array from a row sparse matrix
    and save them in the output sparse matrix.

    Example::

      data = [[1, 2], [3, 4], [5, 6]]
      indices = [0, 1, 3]
      shape = (4, 2)
      rsp_in = row_sparse_array(data, indices)
      to_retain = [0, 3]
      rsp_out = retain(rsp_in, to_retain)
      rsp_out.data = [[1, 2], [5, 6]]
      rsp_out.indices = [0, 3]

    The storage type of ``retain`` output depends on storage types of inputs

    - retain(row_sparse, default) = row_sparse
    - otherwise, ``retain`` is not supported



    Defined in ../src/operator/tensor/sparse_retain.cc:L53

    Parameters
    ----------
    data : NDArray
        The input array for sparse_retain operator.
    indices : NDArray
        The index array of rows ids that will be retained.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)