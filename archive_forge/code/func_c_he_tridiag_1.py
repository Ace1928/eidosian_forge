from ..libmp.backend import xrange
from .eigen import defun
def c_he_tridiag_1(ctx, A, T):
    """
    This routine forms the unitary matrix Q described in c_he_tridiag_0.

    parameters:
      A    (input/output) On input, A is the same matrix as delivered by
           c_he_tridiag_0. On output, A is set to Q.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

    """
    n = A.rows
    for i in xrange(0, n):
        if A[i, i] != 0:
            for j in xrange(0, i):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i, k]) * A[k, j]
                for k in xrange(0, i):
                    A[k, j] -= G * A[k, i]
        A[i, i] = 1
        for j in xrange(0, i):
            A[j, i] = A[i, j] = 0
    for i in xrange(0, n):
        for k in xrange(0, n):
            A[i, k] *= T[k]