from ..libmp.backend import xrange
from .eigen import defun
def svd_c_raw(ctx, A, V=False, calc_u=False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

                    A = U S V

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    The diagonal elements of S are the singular values of A, i.e. the
    squareroots of the eigenvalues of A' A or A A'. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation). Householder bidiagonalization
    and a variant of the QR algorithm is used.

    overview of the matrices :

      A : m*n       A gets replaced by U
      U : m*n       U replaces A. If n>m then only the first m*m block of U is
                    non-zero. column-unitary: U' U = B
                    here B is a n*n matrix whose first min(m,n) diagonal
                    elements are 1 and all other elements are zero.
      S : n*n       diagonal matrix, only the diagonal elements are stored in
                    the array S. only the first min(m,n) diagonal elements are non-zero.
      V : n*n       unitary: V V' = V' V = 1

    parameters:
      A        (input/output) On input, A contains a complex matrix of shape m*n.
               On output, if calc_u is true A contains the column-unitary
               matrix U; otherwise A is simply used as workspace and thus destroyed.

      V        (input/output) if false, the matrix V is not calculated. otherwise
               V must be a matrix of shape n*n.

      calc_u   (input) If true, the matrix U is calculated and replaces A.
               if false, U is not calculated and A is simply destroyed

    return value:
      S        an array of length n containing the singular values of A sorted by
               decreasing magnitude. only the first min(m,n) elements are non-zero.

    This routine is a python translation of the fortran routine svd.f in the
    software library EISPACK (see netlib.org) which itself is based on the
    algol procedure svd described in:
      - num. math. 14, 403-420(1970) by golub and reinsch.
      - wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).

    """
    m, n = (A.rows, A.cols)
    S = ctx.zeros(n, 1)
    work = ctx.zeros(n, 1)
    lbeta = ctx.zeros(n, 1)
    rbeta = ctx.zeros(n, 1)
    dwork = ctx.zeros(n, 1)
    g = scale = anorm = 0
    maxits = 3 * ctx.dps
    for i in xrange(n):
        dwork[i] = scale * g
        g = s = scale = 0
        if i < m:
            for k in xrange(i, m):
                scale += ctx.fabs(ctx.re(A[k, i])) + ctx.fabs(ctx.im(A[k, i]))
            if scale != 0:
                for k in xrange(i, m):
                    A[k, i] /= scale
                    ar = ctx.re(A[k, i])
                    ai = ctx.im(A[k, i])
                    s += ar * ar + ai * ai
                f = A[i, i]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)
                beta /= ctx.conj(beta)
                beta += 1
                h = 2 * (ctx.re(f) * g - s)
                A[i, i] = f - g
                beta /= h
                lbeta[i] = beta / scale / scale
                for j in xrange(i + 1, n):
                    s = 0
                    for k in xrange(i, m):
                        s += ctx.conj(A[k, i]) * A[k, j]
                    f = beta * s
                    for k in xrange(i, m):
                        A[k, j] += f * A[k, i]
                for k in xrange(i, m):
                    A[k, i] *= scale
        S[i] = scale * g
        g = s = scale = 0
        if i < m and i != n - 1:
            for k in xrange(i + 1, n):
                scale += ctx.fabs(ctx.re(A[i, k])) + ctx.fabs(ctx.im(A[i, k]))
            if scale:
                for k in xrange(i + 1, n):
                    A[i, k] /= scale
                    ar = ctx.re(A[i, k])
                    ai = ctx.im(A[i, k])
                    s += ar * ar + ai * ai
                f = A[i, i + 1]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)
                beta /= ctx.conj(beta)
                beta += 1
                h = 2 * (ctx.re(f) * g - s)
                A[i, i + 1] = f - g
                beta /= h
                rbeta[i] = beta / scale / scale
                for k in xrange(i + 1, n):
                    work[k] = A[i, k]
                for j in xrange(i + 1, m):
                    s = 0
                    for k in xrange(i + 1, n):
                        s += ctx.conj(A[i, k]) * A[j, k]
                    f = s * beta
                    for k in xrange(i + 1, n):
                        A[j, k] += f * work[k]
                for k in xrange(i + 1, n):
                    A[i, k] *= scale
        anorm = max(anorm, ctx.fabs(S[i]) + ctx.fabs(dwork[i]))
    if not isinstance(V, bool):
        for i in xrange(n - 2, -1, -1):
            V[i + 1, i + 1] = 1
            if dwork[i + 1] != 0:
                f = ctx.conj(rbeta[i])
                for j in xrange(i + 1, n):
                    V[i, j] = A[i, j] * f
                for j in xrange(i + 1, n):
                    s = 0
                    for k in xrange(i + 1, n):
                        s += ctx.conj(A[i, k]) * V[j, k]
                    for k in xrange(i + 1, n):
                        V[j, k] += s * V[i, k]
            for j in xrange(i + 1, n):
                V[j, i] = V[i, j] = 0
        V[0, 0] = 1
    if m < n:
        minnm = m
    else:
        minnm = n
    if calc_u:
        for i in xrange(minnm - 1, -1, -1):
            g = S[i]
            for j in xrange(i + 1, n):
                A[i, j] = 0
            if g != 0:
                g = 1 / g
                for j in xrange(i + 1, n):
                    s = 0
                    for k in xrange(i + 1, m):
                        s += ctx.conj(A[k, i]) * A[k, j]
                    f = s * ctx.conj(lbeta[i])
                    for k in xrange(i, m):
                        A[k, j] += f * A[k, i]
                for j in xrange(i, m):
                    A[j, i] *= g
            else:
                for j in xrange(i, m):
                    A[j, i] = 0
            A[i, i] += 1
    for k in xrange(n - 1, -1, -1):
        its = 0
        while 1:
            its += 1
            flag = True
            for l in xrange(k, -1, -1):
                nm = l - 1
                if ctx.fabs(dwork[l]) + anorm == anorm:
                    flag = False
                    break
                if ctx.fabs(S[nm]) + anorm == anorm:
                    break
            if flag:
                c = 0
                s = 1
                for i in xrange(l, k + 1):
                    f = s * dwork[i]
                    dwork[i] *= c
                    if ctx.fabs(f) + anorm == anorm:
                        break
                    g = S[i]
                    h = ctx.hypot(f, g)
                    S[i] = h
                    h = 1 / h
                    c = g * h
                    s = -f * h
                    if calc_u:
                        for j in xrange(m):
                            y = A[j, nm]
                            z = A[j, i]
                            A[j, nm] = y * c + z * s
                            A[j, i] = z * c - y * s
            z = S[k]
            if l == k:
                if z < 0:
                    S[k] = -z
                    if not isinstance(V, bool):
                        for j in xrange(n):
                            V[k, j] = -V[k, j]
                break
            if its >= maxits:
                raise RuntimeError('svd: no convergence to an eigenvalue after %d iterations' % its)
            x = S[l]
            nm = k - 1
            y = S[nm]
            g = dwork[nm]
            h = dwork[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
            g = ctx.hypot(f, 1)
            if f >= 0:
                f = ((x - z) * (x + z) + h * (y / (f + g) - h)) / x
            else:
                f = ((x - z) * (x + z) + h * (y / (f - g) - h)) / x
            c = s = 1
            for j in xrange(l, nm + 1):
                g = dwork[j + 1]
                y = S[j + 1]
                h = s * g
                g = c * g
                z = ctx.hypot(f, h)
                dwork[j] = z
                c = f / z
                s = h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                if not isinstance(V, bool):
                    for jj in xrange(n):
                        x = V[j, jj]
                        z = V[j + 1, jj]
                        V[j, jj] = x * c + z * s
                        V[j + 1, jj] = z * c - x * s
                z = ctx.hypot(f, h)
                S[j] = z
                if z != 0:
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                x = c * y - s * g
                if calc_u:
                    for jj in xrange(m):
                        y = A[jj, j]
                        z = A[jj, j + 1]
                        A[jj, j] = y * c + z * s
                        A[jj, j + 1] = z * c - y * s
            dwork[l] = 0
            dwork[k] = f
            S[k] = x
    for i in xrange(n):
        imax = i
        s = ctx.fabs(S[i])
        for j in xrange(i + 1, n):
            c = ctx.fabs(S[j])
            if c > s:
                s = c
                imax = j
        if imax != i:
            z = S[i]
            S[i] = S[imax]
            S[imax] = z
            if calc_u:
                for j in xrange(m):
                    z = A[j, i]
                    A[j, i] = A[j, imax]
                    A[j, imax] = z
            if not isinstance(V, bool):
                for j in xrange(n):
                    z = V[i, j]
                    V[i, j] = V[imax, j]
                    V[imax, j] = z
    return S