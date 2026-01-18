from ..libmp.backend import xrange
from .eigen import defun
@defun
def gauss_quadrature(ctx, n, qtype='legendre', alpha=0, beta=0):
    """
    This routine calulates gaussian quadrature rules for different
    families of orthogonal polynomials. Let (a, b) be an interval,
    W(x) a positive weight function and n a positive integer.
    Then the purpose of this routine is to calculate pairs (x_k, w_k)
    for k=0, 1, 2, ... (n-1) which give

      int(W(x) * F(x), x = a..b) = sum(w_k * F(x_k),k = 0..(n-1))

    exact for all polynomials F(x) of degree (strictly) less than 2*n. For all
    integrable functions F(x) the sum is a (more or less) good approximation to
    the integral. The x_k are called nodes (which are the zeros of the
    related orthogonal polynomials) and the w_k are called the weights.

    parameters
       n        (input) The degree of the quadrature rule, i.e. its number of
                nodes.

       qtype    (input) The family of orthogonal polynmomials for which to
                compute the quadrature rule. See the list below.

       alpha    (input) real number, used as parameter for some orthogonal
                polynomials

       beta     (input) real number, used as parameter for some orthogonal
                polynomials.

    return value

      (X, W)    a pair of two real arrays where x_k = X[k] and w_k = W[k].


    orthogonal polynomials:

      qtype           polynomial
      -----           ----------

      "legendre"      Legendre polynomials, W(x)=1 on the interval (-1, +1)
      "legendre01"    shifted Legendre polynomials, W(x)=1 on the interval (0, +1)
      "hermite"       Hermite polynomials, W(x)=exp(-x*x) on (-infinity,+infinity)
      "laguerre"      Laguerre polynomials, W(x)=exp(-x) on (0,+infinity)
      "glaguerre"     generalized Laguerre polynomials, W(x)=exp(-x)*x**alpha
                      on (0, +infinity)
      "chebyshev1"    Chebyshev polynomials of the first kind, W(x)=1/sqrt(1-x*x)
                      on (-1, +1)
      "chebyshev2"    Chebyshev polynomials of the second kind, W(x)=sqrt(1-x*x)
                      on (-1, +1)
      "jacobi"        Jacobi polynomials, W(x)=(1-x)**alpha * (1+x)**beta on (-1, +1)
                      with alpha>-1 and beta>-1

    examples:
      >>> from mpmath import mp
      >>> f = lambda x: x**8 + 2 * x**6 - 3 * x**4 + 5 * x**2 - 7
      >>> X, W = mp.gauss_quadrature(5, "hermite")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = mp.sqrt(mp.pi) * 57 / 16
      >>> C = mp.quad(lambda x: mp.exp(- x * x) * f(x), [-mp.inf, +mp.inf])
      >>> mp.nprint((mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10)))
      (0.0, 0.0)

      >>> f = lambda x: x**5 - 2 * x**4 + 3 * x**3 - 5 * x**2 + 7 * x - 11
      >>> X, W = mp.gauss_quadrature(3, "laguerre")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = 76
      >>> C = mp.quad(lambda x: mp.exp(-x) * f(x), [0, +mp.inf])
      >>> mp.nprint(mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10))
      .0

      # orthogonality of the chebyshev polynomials:
      >>> f = lambda x: mp.chebyt(3, x) * mp.chebyt(2, x)
      >>> X, W = mp.gauss_quadrature(3, "chebyshev1")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> print(mp.chop(A, tol = 1e-10))
      0.0

    references:
      - golub and welsch, "calculations of gaussian quadrature rules", mathematics of
        computation 23, p. 221-230 (1969)
      - golub, "some modified matrix eigenvalue problems", siam review 15, p. 318-334 (1973)
      - stroud and secrest, "gaussian quadrature formulas", prentice-hall (1966)

    See also the routine gaussq.f in netlog.org or ACM Transactions on
    Mathematical Software algorithm 726.
    """
    d = ctx.zeros(n, 1)
    e = ctx.zeros(n, 1)
    z = ctx.zeros(1, n)
    z[0, 0] = 1
    if qtype == 'legendre':
        w = 2
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j * j / (4 * j * j - ctx.mpf(1)))
    elif qtype == 'legendre01':
        w = 1
        for i in xrange(n):
            d[i] = 1 / ctx.mpf(2)
            j = i + 1
            e[i] = ctx.sqrt(j * j / (16 * j * j - ctx.mpf(4)))
    elif qtype == 'hermite':
        w = ctx.sqrt(ctx.pi)
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j / ctx.mpf(2))
    elif qtype == 'laguerre':
        w = 1
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1
            e[i] = j
    elif qtype == 'chebyshev1':
        w = ctx.pi
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
        e[0] = ctx.sqrt(1 / ctx.mpf(2))
    elif qtype == 'chebyshev2':
        w = ctx.pi / 2
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
    elif qtype == 'glaguerre':
        w = ctx.gamma(1 + alpha)
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1 + alpha
            e[i] = ctx.sqrt(j * (j + alpha))
    elif qtype == 'jacobi':
        alpha = ctx.mpf(alpha)
        beta = ctx.mpf(beta)
        ab = alpha + beta
        abi = ab + 2
        w = 2 ** (ab + 1) * ctx.gamma(alpha + 1) * ctx.gamma(beta + 1) / ctx.gamma(abi)
        d[0] = (beta - alpha) / abi
        e[0] = ctx.sqrt(4 * (1 + alpha) * (1 + beta) / ((abi + 1) * (abi * abi)))
        a2b2 = beta * beta - alpha * alpha
        for i in xrange(1, n):
            j = i + 1
            abi = 2 * j + ab
            d[i] = a2b2 / ((abi - 2) * abi)
            e[i] = ctx.sqrt(4 * j * (j + alpha) * (j + beta) * (j + ab) / ((abi * abi - 1) * abi * abi))
    elif isinstance(qtype, str):
        raise ValueError('unknown quadrature rule "%s"' % qtype)
    elif not isinstance(qtype, str):
        w = qtype(d, e)
    else:
        assert 0
    tridiag_eigen(ctx, d, e, z)
    for i in xrange(len(z)):
        z[i] *= z[i]
    z = z.transpose()
    return (d, w * z)