from ._internal import NDArrayBase
from ..base import _Null
def syevd(A=None, out=None, name=None, **kwargs):
    """Eigendecomposition for symmetric matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, *A* must be symmetric, of shape *(x, x)*. We compute the eigendecomposition,
    resulting in the orthonormal matrix *U* of eigenvectors, shape *(x, x)*, and the
    vector *L* of eigenvalues, shape *(x,)*, so that:

       *U* \\* *A* = *diag(L)* \\* *U*

    Here:

       *U* \\* *U*\\ :sup:`T` = *U*\\ :sup:`T` \\* *U* = *I*

    where *I* is the identity matrix. Also, *L(0) <= L(1) <= L(2) <= ...* (ascending order).

    If *n>2*, *syevd* is performed separately on the trailing two dimensions of *A* (batch
    mode). In this case, *U* has *n* dimensions like *A*, and *L* has *n-1* dimensions.

    .. note:: The operator supports float32 and float64 data types only.

    .. note:: Derivatives for this operator are defined only if *A* is such that all its
              eigenvalues are distinct, and the eigengaps are not too small. If you need
              gradients, do not apply this operator to matrices with multiple eigenvalues.

    Examples::

       Single symmetric eigendecomposition
       A = [[1., 2.], [2., 4.]]
       U, L = syevd(A)
       U = [[0.89442719, -0.4472136],
            [0.4472136, 0.89442719]]
       L = [0., 5.]

       Batch symmetric eigendecomposition
       A = [[[1., 2.], [2., 4.]],
            [[1., 2.], [2., 5.]]]
       U, L = syevd(A)
       U = [[[0.89442719, -0.4472136],
             [0.4472136, 0.89442719]],
            [[0.92387953, -0.38268343],
             [0.38268343, 0.92387953]]]
       L = [[0., 5.],
            [0.17157288, 5.82842712]]


    Defined in ../src/operator/tensor/la_op.cc:L867

    Parameters
    ----------
    A : NDArray
        Tensor of input matrices to be factorized

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)