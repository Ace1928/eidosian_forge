from ..libmp.backend import xrange
from .eigen import defun
@defun
def eighe(ctx, A, eigvals_only=False, overwrite_a=False):
    """
    This routine solves the (ordinary) eigenvalue problem for a complex
    hermitian square matrix A. Given A, an unitary matrix Q is calculated which
    diagonalizes A:

        Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) a is diagonal matrix whose diagonal is E.
    ' denotes the hermitian transpose (i.e. ordinary transposition and
    complex conjugation).

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

        A Q[:,i] = E[i] Q[:,i]


    input:

      A: complex matrix of format (n,n) which is hermitian
         (i.e. A=A' or A[i,j]=conj(A[j,i]))

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: unitary matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

           E         if eigvals_only is true
          (E, Q)     if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[1, -3 - 1j], [-3 + 1j, -2]])
      >>> E = mp.eighe(A, eigvals_only = True)
      >>> print(E)
      [-4.0]
      [ 3.0]

      >>> A = mp.matrix([[1, 2 + 5j], [2 - 5j, 3]])
      >>> E, Q = mp.eighe(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eigsy, eigh, eig
    """
    if not overwrite_a:
        A = A.copy()
    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)
    t = ctx.zeros(A.rows, 1)
    if eigvals_only:
        c_he_tridiag_0(ctx, A, d, e, t)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        c_he_tridiag_0(ctx, A, d, e, t)
        B = ctx.eye(A.rows)
        tridiag_eigen(ctx, d, e, B)
        c_he_tridiag_2(ctx, A, t, B)
        return (d, B)