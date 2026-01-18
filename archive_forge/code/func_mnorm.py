from ..libmp.backend import xrange
import warnings
def mnorm(ctx, A, p=1):
    """
        Gives the matrix (operator) `p`-norm of A. Currently ``p=1`` and ``p=inf``
        are supported:

        ``p=1`` gives the 1-norm (maximal column sum)

        ``p=inf`` gives the `\\infty`-norm (maximal row sum).
        You can use the string 'inf' as well as float('inf') or mpf('inf')

        ``p=2`` (not implemented) for a square matrix is the usual spectral
        matrix norm, i.e. the largest singular value.

        ``p='f'`` (or 'F', 'fro', 'Frobenius, 'frobenius') gives the
        Frobenius norm, which is the elementwise 2-norm. The Frobenius norm is an
        approximation of the spectral norm and satisfies

        .. math ::

            \\frac{1}{\\sqrt{\\mathrm{rank}(A)}} \\|A\\|_F \\le \\|A\\|_2 \\le \\|A\\|_F

        The Frobenius norm lacks some mathematical properties that might
        be expected of a norm.

        For general elementwise `p`-norms, use :func:`~mpmath.norm` instead.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = matrix([[1, -1000], [100, 50]])
            >>> mnorm(A, 1)
            mpf('1050.0')
            >>> mnorm(A, inf)
            mpf('1001.0')
            >>> mnorm(A, 'F')
            mpf('1006.2310867787777')

        """
    A = ctx.matrix(A)
    if type(p) is not int:
        if type(p) is str and 'frobenius'.startswith(p.lower()):
            return ctx.norm(A, 2)
        p = ctx.convert(p)
    m, n = (A.rows, A.cols)
    if p == 1:
        return max((ctx.fsum((A[i, j] for i in xrange(m)), absolute=1) for j in xrange(n)))
    elif p == ctx.inf:
        return max((ctx.fsum((A[i, j] for j in xrange(n)), absolute=1) for i in xrange(m)))
    else:
        raise NotImplementedError('matrix p-norm for arbitrary p')