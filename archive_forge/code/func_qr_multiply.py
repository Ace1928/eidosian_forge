import numpy
from .lapack import get_lapack_funcs
from ._misc import _datacopied
def qr_multiply(a, c, mode='right', pivoting=False, conjugate=False, overwrite_a=False, overwrite_c=False):
    """
    Calculate the QR decomposition and multiply Q with a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular. Multiply Q with a vector or a matrix c.

    Parameters
    ----------
    a : (M, N), array_like
        Input array
    c : array_like
        Input array to be multiplied by ``q``.
    mode : {'left', 'right'}, optional
        ``Q @ c`` is returned if mode is 'left', ``c @ Q`` is returned if
        mode is 'right'.
        The shape of c must be appropriate for the matrix multiplications,
        if mode is 'left', ``min(a.shape) == c.shape[0]``,
        if mode is 'right', ``a.shape[0] == c.shape[1]``.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition, see the documentation of qr.
    conjugate : bool, optional
        Whether Q should be complex-conjugated. This might be faster
        than explicit conjugation.
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    overwrite_c : bool, optional
        Whether data in c is overwritten (may improve performance).
        If this is used, c must be big enough to keep the result,
        i.e. ``c.shape[0]`` = ``a.shape[0]`` if mode is 'left'.

    Returns
    -------
    CQ : ndarray
        The product of ``Q`` and ``c``.
    R : (K, N), ndarray
        R array of the resulting QR factorization where ``K = min(M, N)``.
    P : (N,) ndarray
        Integer pivot array. Only returned when ``pivoting=True``.

    Raises
    ------
    LinAlgError
        Raised if QR decomposition fails.

    Notes
    -----
    This is an interface to the LAPACK routines ``?GEQRF``, ``?ORMQR``,
    ``?UNMQR``, and ``?GEQP3``.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import qr_multiply, qr
    >>> A = np.array([[1, 3, 3], [2, 3, 2], [2, 3, 3], [1, 3, 2]])
    >>> qc, r1, piv1 = qr_multiply(A, 2*np.eye(4), pivoting=1)
    >>> qc
    array([[-1.,  1., -1.],
           [-1., -1.,  1.],
           [-1., -1., -1.],
           [-1.,  1.,  1.]])
    >>> r1
    array([[-6., -3., -5.            ],
           [ 0., -1., -1.11022302e-16],
           [ 0.,  0., -1.            ]])
    >>> piv1
    array([1, 0, 2], dtype=int32)
    >>> q2, r2, piv2 = qr(A, mode='economic', pivoting=1)
    >>> np.allclose(2*q2 - qc, np.zeros((4, 3)))
    True

    """
    if mode not in ['left', 'right']:
        raise ValueError(f"Mode argument can only be 'left' or 'right' but not '{mode}'")
    c = numpy.asarray_chkfinite(c)
    if c.ndim < 2:
        onedim = True
        c = numpy.atleast_2d(c)
        if mode == 'left':
            c = c.T
    else:
        onedim = False
    a = numpy.atleast_2d(numpy.asarray(a))
    M, N = a.shape
    if mode == 'left':
        if c.shape[0] != min(M, N + overwrite_c * (M - N)):
            raise ValueError(f'Array shapes are not compatible for Q @ c operation: {a.shape} vs {c.shape}')
    elif M != c.shape[1]:
        raise ValueError(f'Array shapes are not compatible for c @ Q operation: {c.shape} vs {a.shape}')
    raw = qr(a, overwrite_a, None, 'raw', pivoting)
    Q, tau = raw[0]
    gor_un_mqr, = get_lapack_funcs(('ormqr',), (Q,))
    if gor_un_mqr.typecode in ('s', 'd'):
        trans = 'T'
    else:
        trans = 'C'
    Q = Q[:, :min(M, N)]
    if M > N and mode == 'left' and (not overwrite_c):
        if conjugate:
            cc = numpy.zeros((c.shape[1], M), dtype=c.dtype, order='F')
            cc[:, :N] = c.T
        else:
            cc = numpy.zeros((M, c.shape[1]), dtype=c.dtype, order='F')
            cc[:N, :] = c
            trans = 'N'
        if conjugate:
            lr = 'R'
        else:
            lr = 'L'
        overwrite_c = True
    elif c.flags['C_CONTIGUOUS'] and trans == 'T' or conjugate:
        cc = c.T
        if mode == 'left':
            lr = 'R'
        else:
            lr = 'L'
    else:
        trans = 'N'
        cc = c
        if mode == 'left':
            lr = 'L'
        else:
            lr = 'R'
    cQ, = safecall(gor_un_mqr, 'gormqr/gunmqr', lr, trans, Q, tau, cc, overwrite_c=overwrite_c)
    if trans != 'N':
        cQ = cQ.T
    if mode == 'right':
        cQ = cQ[:, :min(M, N)]
    if onedim:
        cQ = cQ.ravel()
    return (cQ,) + raw[1:]