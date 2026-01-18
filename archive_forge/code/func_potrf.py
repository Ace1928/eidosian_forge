from ._internal import NDArrayBase
from ..base import _Null
def potrf(A=None, out=None, name=None, **kwargs):
    """Performs Cholesky factorization of a symmetric positive-definite matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, the Cholesky factor *B* of the symmetric, positive definite matrix *A* is
    computed. *B* is triangular (entries of upper or lower triangle are all zero), has
    positive diagonal entries, and:

      *A* = *B* \\* *B*\\ :sup:`T`  if *lower* = *true*
      *A* = *B*\\ :sup:`T` \\* *B*  if *lower* = *false*

    If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs
    (batch mode).

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

       Single matrix factorization
       A = [[4.0, 1.0], [1.0, 4.25]]
       potrf(A) = [[2.0, 0], [0.5, 2.0]]

       Batch matrix factorization
       A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
       potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]


    Defined in ../src/operator/tensor/la_op.cc:L213

    Parameters
    ----------
    A : NDArray
        Tensor of input matrices to be decomposed

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)