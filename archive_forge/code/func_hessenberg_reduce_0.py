from ..libmp.backend import xrange
def hessenberg_reduce_0(ctx, A, T):
    """
    This routine computes the (upper) Hessenberg decomposition of a square matrix A.
    Given A, an unitary matrix Q is calculated such that

               Q' A Q = H              and             Q' Q = Q Q' = 1

    where H is an upper Hessenberg matrix, meaning that it only contains zeros
    below the first subdiagonal. Here ' denotes the hermitian transpose (i.e.
    transposition and conjugation).

    parameters:
      A         (input/output) On input, A contains the square matrix A of
                dimension (n,n). On output, A contains a compressed representation
                of Q and H.
      T         (output) An array of length n containing the first elements of
                the Householder reflectors.
    """
    n = A.rows
    if n <= 2:
        return
    for i in xrange(n - 1, 1, -1):
        scale = 0
        for k in xrange(0, i):
            scale += abs(ctx.re(A[i, k])) + abs(ctx.im(A[i, k]))
        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale
        if scale == 0 or ctx.isinf(scale_inv):
            T[i] = 0
            A[i, i - 1] = 0
            continue
        H = 0
        for k in xrange(0, i):
            A[i, k] *= scale_inv
            rr = ctx.re(A[i, k])
            ii = ctx.im(A[i, k])
            H += rr * rr + ii * ii
        F = A[i, i - 1]
        f = abs(F)
        G = ctx.sqrt(H)
        A[i, i - 1] = -G * scale
        if f == 0:
            T[i] = G
        else:
            ff = F / f
            T[i] = F + G * ff
            A[i, i - 1] *= ff
        H += G * f
        H = 1 / ctx.sqrt(H)
        T[i] *= H
        for k in xrange(0, i - 1):
            A[i, k] *= H
        for j in xrange(0, i):
            G = ctx.conj(T[i]) * A[j, i - 1]
            for k in xrange(0, i - 1):
                G += ctx.conj(A[i, k]) * A[j, k]
            A[j, i - 1] -= G * T[i]
            for k in xrange(0, i - 1):
                A[j, k] -= G * A[i, k]
        for j in xrange(0, n):
            G = T[i] * A[i - 1, j]
            for k in xrange(0, i - 1):
                G += A[i, k] * A[k, j]
            A[i - 1, j] -= G * ctx.conj(T[i])
            for k in xrange(0, i - 1):
                A[k, j] -= G * ctx.conj(A[i, k])