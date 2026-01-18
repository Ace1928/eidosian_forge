from ..libmp.backend import xrange
from .eigen import defun
def r_sy_tridiag(ctx, A, D, E, calc_ev=True):
    """
    This routine transforms a real symmetric matrix A to a real symmetric
    tridiagonal matrix T using an orthogonal similarity transformation:
          Q' * A * Q = T     (here ' denotes the matrix transpose).
    The orthogonal matrix Q is build up from Householder reflectors.

    parameters:
      A         (input/output) On input, A contains the real symmetric matrix of
                dimension (n,n). On output, if calc_ev is true, A contains the
                orthogonal matrix Q, otherwise A is destroyed.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      calc_ev   (input) If calc_ev is true, this routine explicitly calculates the
                orthogonal matrix Q which is then returned in A. If calc_ev is
                false, Q is not explicitly calculated resulting in a shorter run time.

    This routine is a python translation of the fortran routine tred2.f in the
    software library EISPACK (see netlib.org) which itself is based on the algol
    procedure tred2 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """
    n = A.rows
    for i in xrange(n - 1, 0, -1):
        scale = 0
        for k in xrange(0, i):
            scale += abs(A[k, i])
        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale
        if i == 1 or scale == 0 or ctx.isinf(scale_inv):
            E[i] = A[i - 1, i]
            D[i] = 0
            continue
        H = 0
        for k in xrange(0, i):
            A[k, i] *= scale_inv
            H += A[k, i] * A[k, i]
        F = A[i - 1, i]
        G = ctx.sqrt(H)
        if F > 0:
            G = -G
        E[i] = scale * G
        H -= F * G
        A[i - 1, i] = F - G
        F = 0
        for j in xrange(0, i):
            if calc_ev:
                A[i, j] = A[j, i] / H
            G = 0
            for k in xrange(0, j + 1):
                G += A[k, j] * A[k, i]
            for k in xrange(j + 1, i):
                G += A[j, k] * A[k, i]
            E[j] = G / H
            F += E[j] * A[j, i]
        HH = F / (2 * H)
        for j in xrange(0, i):
            F = A[j, i]
            G = E[j] - HH * F
            E[j] = G
            for k in xrange(0, j + 1):
                A[k, j] -= F * E[k] + G * A[k, i]
        D[i] = H
    for i in xrange(1, n):
        E[i - 1] = E[i]
    E[n - 1] = 0
    if calc_ev:
        D[0] = 0
        for i in xrange(0, n):
            if D[i] != 0:
                for j in xrange(0, i):
                    G = 0
                    for k in xrange(0, i):
                        G += A[i, k] * A[k, j]
                    for k in xrange(0, i):
                        A[k, j] -= G * A[k, i]
            D[i] = A[i, i]
            A[i, i] = 1
            for j in xrange(0, i):
                A[j, i] = A[i, j] = 0
    else:
        for i in xrange(0, n):
            D[i] = A[i, i]