from ._internal import NDArrayBase
from ..base import _Null
def linalg_inverse(A=None, out=None, name=None, **kwargs):
    """Compute the inverse of a matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, *A* is a square matrix. We compute:

      *out* = *A*\\ :sup:`-1`

    If *n>2*, *inverse* is performed separately on the trailing two dimensions
    for all inputs (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       Single matrix inverse
       A = [[1., 4.], [2., 3.]]
       inverse(A) = [[-0.6, 0.8], [0.4, -0.2]]

       Batch matrix inverse
       A = [[[1., 4.], [2., 3.]],
            [[1., 3.], [2., 4.]]]
       inverse(A) = [[[-0.6, 0.8], [0.4, -0.2]],
                     [[-2., 1.5], [1., -0.5]]]


    Defined in ../src/operator/tensor/la_op.cc:L919

    Parameters
    ----------
    A : NDArray
        Tensor of square matrix

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)