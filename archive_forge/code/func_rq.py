import numpy
from .lapack import get_lapack_funcs
from ._misc import _datacopied
def rq(a, overwrite_a=False, lwork=None, mode='full', check_finite=True):
    """
    Compute RQ decomposition of a matrix.

    Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    lwork : int, optional
        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
        is computed.
    mode : {'full', 'r', 'economic'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : float or complex ndarray
        Of shape (M, N) or (M, K) for ``mode='economic'``. ``K = min(M, N)``.
    Q : float or complex ndarray
        Of shape (N, N) or (K, N) for ``mode='economic'``. Not returned
        if ``mode='r'``.

    Raises
    ------
    LinAlgError
        If decomposition fails.

    Notes
    -----
    This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,
    sorgrq, dorgrq, cungrq and zungrq.

    If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead
    of (N,N) and (M,N), with ``K=min(M,N)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((6, 9))
    >>> r, q = linalg.rq(a)
    >>> np.allclose(a, r @ q)
    True
    >>> r.shape, q.shape
    ((6, 9), (9, 9))
    >>> r2 = linalg.rq(a, mode='r')
    >>> np.allclose(r, r2)
    True
    >>> r3, q3 = linalg.rq(a, mode='economic')
    >>> r3.shape, q3.shape
    ((6, 6), (6, 9))

    """
    if mode not in ['full', 'r', 'economic']:
        raise ValueError("Mode argument should be one of ['full', 'r', 'economic']")
    if check_finite:
        a1 = numpy.asarray_chkfinite(a)
    else:
        a1 = numpy.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    M, N = a1.shape
    overwrite_a = overwrite_a or _datacopied(a1, a)
    gerqf, = get_lapack_funcs(('gerqf',), (a1,))
    rq, tau = safecall(gerqf, 'gerqf', a1, lwork=lwork, overwrite_a=overwrite_a)
    if not mode == 'economic' or N < M:
        R = numpy.triu(rq, N - M)
    else:
        R = numpy.triu(rq[-M:, -M:])
    if mode == 'r':
        return R
    gor_un_grq, = get_lapack_funcs(('orgrq',), (rq,))
    if N < M:
        Q, = safecall(gor_un_grq, 'gorgrq/gungrq', rq[-N:], tau, lwork=lwork, overwrite_a=1)
    elif mode == 'economic':
        Q, = safecall(gor_un_grq, 'gorgrq/gungrq', rq, tau, lwork=lwork, overwrite_a=1)
    else:
        rq1 = numpy.empty((N, N), dtype=rq.dtype)
        rq1[-M:] = rq
        Q, = safecall(gor_un_grq, 'gorgrq/gungrq', rq1, tau, lwork=lwork, overwrite_a=1)
    return (R, Q)