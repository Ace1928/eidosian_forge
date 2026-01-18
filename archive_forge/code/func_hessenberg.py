import warnings
import numpy
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def hessenberg(a, calc_q=False, overwrite_a=False, check_finite=True):
    """
    Compute Hessenberg form of a matrix.

    The Hessenberg decomposition is::

        A = Q H Q^H

    where `Q` is unitary/orthogonal and `H` has only zero elements below
    the first sub-diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to bring into Hessenberg form.
    calc_q : bool, optional
        Whether to compute the transformation matrix.  Default is False.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    H : (M, M) ndarray
        Hessenberg form of `a`.
    Q : (M, M) ndarray
        Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.
        Only returned if ``calc_q=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import hessenberg
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> H, Q = hessenberg(A, calc_q=True)
    >>> H
    array([[  2.        , -11.65843866,   1.42005301,   0.25349066],
           [ -9.94987437,  14.53535354,  -5.31022304,   2.43081618],
           [  0.        ,  -1.83299243,   0.38969961,  -0.51527034],
           [  0.        ,   0.        ,  -3.83189513,   1.07494686]])
    >>> np.allclose(Q @ H @ Q.conj().T - A, np.zeros((4, 4)))
    True
    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or _datacopied(a1, a)
    if a1.shape[0] <= 2:
        if calc_q:
            return (a1, eye(a1.shape[0]))
        return a1
    gehrd, gebal, gehrd_lwork = get_lapack_funcs(('gehrd', 'gebal', 'gehrd_lwork'), (a1,))
    ba, lo, hi, pivscale, info = gebal(a1, permute=0, overwrite_a=overwrite_a)
    _check_info(info, 'gebal (hessenberg)', positive=False)
    n = len(a1)
    lwork = _compute_lwork(gehrd_lwork, ba.shape[0], lo=lo, hi=hi)
    hq, tau, info = gehrd(ba, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    _check_info(info, 'gehrd (hessenberg)', positive=False)
    h = numpy.triu(hq, -1)
    if not calc_q:
        return h
    orghr, orghr_lwork = get_lapack_funcs(('orghr', 'orghr_lwork'), (a1,))
    lwork = _compute_lwork(orghr_lwork, n, lo=lo, hi=hi)
    q, info = orghr(a=hq, tau=tau, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    _check_info(info, 'orghr (hessenberg)', positive=False)
    return (h, q)