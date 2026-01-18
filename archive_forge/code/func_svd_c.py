from ..libmp.backend import xrange
from .eigen import defun
@defun
def svd_c(ctx, A, full_matrices=False, compute_uv=True, overwrite_a=False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the hermitian transpose (i.e. transposition and complex
    conjugation). The diagonal elements of S are the singular values of A,
    i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a complex matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an unitary matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an unitary matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[-2j, -1-3j, -2+2j], [2-2j, -1-3j, 1], [-3+1j,-2j,0]])
      >>> S = mp.svd_c(A, compute_uv = False)
      >>> print(mp.chop(S - mp.matrix([mp.sqrt(34), mp.sqrt(15), mp.sqrt(6)])))
      [0.0]
      [0.0]
      [0.0]

      >>> U, S, V = mp.svd_c(A)
      >>> print(mp.chop(A - U * mp.diag(S) * V))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    see also: svd, svd_r
    """
    m, n = (A.rows, A.cols)
    if not compute_uv:
        if not overwrite_a:
            A = A.copy()
        S = svd_c_raw(ctx, A, V=False, calc_u=False)
        S = S[:min(m, n)]
        return S
    if full_matrices and n < m:
        V = ctx.zeros(m, m)
        A0 = ctx.zeros(m, m)
        A0[:, :n] = A
        S = svd_c_raw(ctx, A0, V, calc_u=True)
        S = S[:n]
        V = V[:n, :n]
        return (A0, S, V)
    else:
        if not overwrite_a:
            A = A.copy()
        V = ctx.zeros(n, n)
        S = svd_c_raw(ctx, A, V, calc_u=True)
        if n > m:
            if full_matrices == False:
                V = V[:m, :]
            S = S[:m]
            A = A[:, :m]
        return (A, S, V)