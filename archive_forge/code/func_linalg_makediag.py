from ._internal import NDArrayBase
from ..base import _Null
def linalg_makediag(A=None, offset=_Null, out=None, name=None, **kwargs):
    """Constructs a square matrix with the input as diagonal.
    Input is a tensor *A* of dimension *n >= 1*.

    If *n=1*, then *A* represents the diagonal entries of a single square matrix. This matrix will be returned as a 2-dimensional tensor.
    If *n>1*, then *A* represents a batch of diagonals of square matrices. The batch of diagonal matrices will be returned as an *n+1*-dimensional tensor.

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

        Single diagonal matrix construction
        A = [1.0, 2.0]

        makediag(A)    = [[1.0, 0.0],
                          [0.0, 2.0]]

        makediag(A, 1) = [[0.0, 1.0, 0.0],
                          [0.0, 0.0, 2.0],
                          [0.0, 0.0, 0.0]]

        Batch diagonal matrix construction
        A = [[1.0, 2.0],
             [3.0, 4.0]]

        makediag(A) = [[[1.0, 0.0],
                        [0.0, 2.0]],
                       [[3.0, 0.0],
                        [0.0, 4.0]]]


    Defined in ../src/operator/tensor/la_op.cc:L546

    Parameters
    ----------
    A : NDArray
        Tensor of diagonal entries
    offset : int, optional, default='0'
        Offset of the diagonal versus the main diagonal. 0 corresponds to the main diagonal, a negative/positive value to diagonals below/above the main diagonal.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)