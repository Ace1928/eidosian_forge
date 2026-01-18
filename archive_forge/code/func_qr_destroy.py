from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
def qr_destroy(la):
    """Get QR decomposition of `la[0]`.

    Parameters
    ----------
    la : list of numpy.ndarray
        Run QR decomposition on the first elements of `la`. Must not be empty.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Matrices :math:`Q` and :math:`R`.

    Notes
    -----
    Using this function is less memory intense than calling `scipy.linalg.qr(la[0])`,
    because the memory used in `la[0]` is reclaimed earlier. This makes a difference when
    decomposing very large arrays, where every memory copy counts.

    Warnings
    --------
    Content of `la` as well as `la[0]` gets destroyed in the process. Again, for memory-effiency reasons.

    """
    a = np.asfortranarray(la[0])
    del la[0], la
    m, n = a.shape
    logger.debug('computing QR of %s dense matrix', str(a.shape))
    geqrf, = get_lapack_funcs(('geqrf',), (a,))
    qr, tau, work, info = geqrf(a, lwork=-1, overwrite_a=True)
    qr, tau, work, info = geqrf(a, lwork=work[0], overwrite_a=True)
    del a
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n:
        qr = qr[:, :m]
    gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
    q, work, info = gorgqr(qr, tau, lwork=-1, overwrite_a=True)
    q, work, info = gorgqr(qr, tau, lwork=work[0], overwrite_a=True)
    assert info >= 0, 'qr failed'
    assert q.flags.f_contiguous
    return (q, r)