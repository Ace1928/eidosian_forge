import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
def spilu(A, drop_tol=None, fill_factor=None, drop_rule=None, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options={}):
    """Computes the incomplete LU decomposition of a sparse square matrix.

    Args:
        A (cupyx.scipy.sparse.spmatrix): Sparse matrix to factorize.
        drop_tol (float): (For further augments, see
            :func:`scipy.sparse.linalg.spilu`)
        fill_factor (float):
        drop_rule (str):
        permc_spec (str):
        diag_pivot_thresh (float):
        relax (int):
        panel_size (int):
        options (dict):

    Returns:
        cupyx.scipy.sparse.linalg.SuperLU:
            Object which has a ``solve`` method.

    Note:
        This function computes incomplete LU decomposition of a sparse matrix
        on the CPU using `scipy.sparse.linalg.spilu` (unless you set
        ``fill_factor`` to ``1``). Therefore, incomplete LU decomposition is
        not accelerated on the GPU. On the other hand, the computation of
        solving linear equations using the ``solve`` method, which this
        function returns, is performed on the GPU.

        If you set ``fill_factor`` to ``1``, this function computes incomplete
        LU decomposition on the GPU, but without fill-in or pivoting.

    .. seealso:: :func:`scipy.sparse.linalg.spilu`
    """
    if not scipy_available:
        raise RuntimeError('scipy is not available')
    if not sparse.isspmatrix(A):
        raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix (A.shape: {})'.format(A.shape))
    if A.dtype.char not in 'fdFD':
        raise TypeError('Invalid dtype (actual: {})'.format(A.dtype))
    if fill_factor == 1:
        if not sparse.isspmatrix_csr(A):
            a = A.tocsr()
        else:
            a = A.copy()
        cusparse.csrilu02(a)
        return CusparseLU(a)
    a = A.get().tocsc()
    a_inv = scipy.sparse.linalg.spilu(a, fill_factor=fill_factor, drop_tol=drop_tol, drop_rule=drop_rule, permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh, relax=relax, panel_size=panel_size, options=options)
    return SuperLU(a_inv)