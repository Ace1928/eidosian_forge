from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def hyper2d(ctx, a, b, x, y, **kwargs):
    """
    Sums the generalized 2D hypergeometric series

    .. math ::

        \\sum_{m=0}^{\\infty} \\sum_{n=0}^{\\infty}
            \\frac{P((a),m,n)}{Q((b),m,n)}
            \\frac{x^m y^n} {m! n!}

    where `(a) = (a_1,\\ldots,a_r)`, `(b) = (b_1,\\ldots,b_s)` and where
    `P` and `Q` are products of rising factorials such as `(a_j)_n` or
    `(a_j)_{m+n}`. `P` and `Q` are specified in the form of dicts, with
    the `m` and `n` dependence as keys and parameter lists as values.
    The supported rising factorials are given in the following table
    (note that only a few are supported in `Q`):

    +------------+-------------------+--------+
    | Key        |  Rising factorial | `Q`    |
    +============+===================+========+
    | ``'m'``    |   `(a_j)_m`       | Yes    |
    +------------+-------------------+--------+
    | ``'n'``    |   `(a_j)_n`       | Yes    |
    +------------+-------------------+--------+
    | ``'m+n'``  |   `(a_j)_{m+n}`   | Yes    |
    +------------+-------------------+--------+
    | ``'m-n'``  |   `(a_j)_{m-n}`   | No     |
    +------------+-------------------+--------+
    | ``'n-m'``  |   `(a_j)_{n-m}`   | No     |
    +------------+-------------------+--------+
    | ``'2m+n'`` |   `(a_j)_{2m+n}`  | No     |
    +------------+-------------------+--------+
    | ``'2m-n'`` |   `(a_j)_{2m-n}`  | No     |
    +------------+-------------------+--------+
    | ``'2n-m'`` |   `(a_j)_{2n-m}`  | No     |
    +------------+-------------------+--------+

    For example, the Appell F1 and F4 functions

    .. math ::

        F_1 = \\sum_{m=0}^{\\infty} \\sum_{n=0}^{\\infty}
              \\frac{(a)_{m+n} (b)_m (c)_n}{(d)_{m+n}}
              \\frac{x^m y^n}{m! n!}

        F_4 = \\sum_{m=0}^{\\infty} \\sum_{n=0}^{\\infty}
              \\frac{(a)_{m+n} (b)_{m+n}}{(c)_m (d)_{n}}
              \\frac{x^m y^n}{m! n!}

    can be represented respectively as

        ``hyper2d({'m+n':[a], 'm':[b], 'n':[c]}, {'m+n':[d]}, x, y)``

        ``hyper2d({'m+n':[a,b]}, {'m':[c], 'n':[d]}, x, y)``

    More generally, :func:`~mpmath.hyper2d` can evaluate any of the 34 distinct
    convergent second-order (generalized Gaussian) hypergeometric
    series enumerated by Horn, as well as the Kampe de Feriet
    function.

    The series is computed by rewriting it so that the inner
    series (i.e. the series containing `n` and `y`) has the form of an
    ordinary generalized hypergeometric series and thereby can be
    evaluated efficiently using :func:`~mpmath.hyper`. If possible,
    manually swapping `x` and `y` and the corresponding parameters
    can sometimes give better results.

    **Examples**

    Two separable cases: a product of two geometric series, and a
    product of two Gaussian hypergeometric functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> x, y = mpf(0.25), mpf(0.5)
        >>> hyper2d({'m':1,'n':1}, {}, x,y)
        2.666666666666666666666667
        >>> 1/(1-x)/(1-y)
        2.666666666666666666666667
        >>> hyper2d({'m':[1,2],'n':[3,4]}, {'m':[5],'n':[6]}, x,y)
        4.164358531238938319669856
        >>> hyp2f1(1,2,5,x)*hyp2f1(3,4,6,y)
        4.164358531238938319669856

    Some more series that can be done in closed form::

        >>> hyper2d({'m':1,'n':1},{'m+n':1},x,y)
        2.013417124712514809623881
        >>> (exp(x)*x-exp(y)*y)/(x-y)
        2.013417124712514809623881

    Six of the 34 Horn functions, G1-G3 and H1-H3::

        >>> from mpmath import *
        >>> mp.dps = 10; mp.pretty = True
        >>> x, y = 0.0625, 0.125
        >>> a1,a2,b1,b2,c1,c2,d = 1.1,-1.2,-1.3,-1.4,1.5,-1.6,1.7
        >>> hyper2d({'m+n':a1,'n-m':b1,'m-n':b2},{},x,y)  # G1
        1.139090746
        >>> nsum(lambda m,n: rf(a1,m+n)*rf(b1,n-m)*rf(b2,m-n)*\\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.139090746
        >>> hyper2d({'m':a1,'n':a2,'n-m':b1,'m-n':b2},{},x,y)  # G2
        0.9503682696
        >>> nsum(lambda m,n: rf(a1,m)*rf(a2,n)*rf(b1,n-m)*rf(b2,m-n)*\\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.9503682696
        >>> hyper2d({'2n-m':a1,'2m-n':a2},{},x,y)  # G3
        1.029372029
        >>> nsum(lambda m,n: rf(a1,2*n-m)*rf(a2,2*m-n)*\\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.029372029
        >>> hyper2d({'m-n':a1,'m+n':b1,'n':c1},{'m':d},x,y)  # H1
        -1.605331256
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m+n)*rf(c1,n)/rf(d,m)*\\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -1.605331256
        >>> hyper2d({'m-n':a1,'m':b1,'n':[c1,c2]},{'m':d},x,y)  # H2
        -2.35405404
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m)*rf(c1,n)*rf(c2,n)/rf(d,m)*\\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -2.35405404
        >>> hyper2d({'2m+n':a1,'n':b1},{'m+n':c1},x,y)  # H3
        0.974479074
        >>> nsum(lambda m,n: rf(a1,2*m+n)*rf(b1,n)/rf(c1,m+n)*\\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.974479074

    **References**

    1. [SrivastavaKarlsson]_
    2. [Weisstein]_ http://mathworld.wolfram.com/HornFunction.html
    3. [Weisstein]_ http://mathworld.wolfram.com/AppellHypergeometricFunction.html

    """
    x = ctx.convert(x)
    y = ctx.convert(y)

    def parse(dct, key):
        args = dct.pop(key, [])
        try:
            args = list(args)
        except TypeError:
            args = [args]
        return [ctx.convert(arg) for arg in args]
    a_s = dict(a)
    b_s = dict(b)
    a_m = parse(a, 'm')
    a_n = parse(a, 'n')
    a_m_add_n = parse(a, 'm+n')
    a_m_sub_n = parse(a, 'm-n')
    a_n_sub_m = parse(a, 'n-m')
    a_2m_add_n = parse(a, '2m+n')
    a_2m_sub_n = parse(a, '2m-n')
    a_2n_sub_m = parse(a, '2n-m')
    b_m = parse(b, 'm')
    b_n = parse(b, 'n')
    b_m_add_n = parse(b, 'm+n')
    if a:
        raise ValueError('unsupported key: %r' % a.keys()[0])
    if b:
        raise ValueError('unsupported key: %r' % b.keys()[0])
    s = 0
    outer = ctx.one
    m = ctx.mpf(0)
    ok_count = 0
    prec = ctx.prec
    maxterms = kwargs.get('maxterms', 20 * prec)
    try:
        ctx.prec += 10
        tol = +ctx.eps
        while 1:
            inner_sign = 1
            outer_sign = 1
            inner_a = list(a_n)
            inner_b = list(b_n)
            outer_a = [a + m for a in a_m]
            outer_b = [b + m for b in b_m]
            for a in a_m_add_n:
                a = a + m
                inner_a.append(a)
                outer_a.append(a)
            for b in b_m_add_n:
                b = b + m
                inner_b.append(b)
                outer_b.append(b)
            for a in a_n_sub_m:
                inner_a.append(a - m)
                outer_b.append(a - m - 1)
            for a in a_m_sub_n:
                inner_sign *= -1
                outer_sign *= (-1) ** m
                inner_b.append(1 - a - m)
                outer_a.append(-a - m)
            for a in a_2m_add_n:
                inner_a.append(a + 2 * m)
                outer_a.append((a + 2 * m) * (1 + a + 2 * m))
            for a in a_2m_sub_n:
                inner_sign *= -1
                inner_b.append(1 - a - 2 * m)
                outer_a.append((a + 2 * m) * (1 + a + 2 * m))
            for a in a_2n_sub_m:
                inner_sign *= 4
                inner_a.append(0.5 * (a - m))
                inner_a.append(0.5 * (a - m + 1))
                outer_b.append(a - m - 1)
            inner = ctx.hyper(inner_a, inner_b, inner_sign * y, zeroprec=ctx.prec, **kwargs)
            term = outer * inner * outer_sign
            if abs(term) < tol:
                ok_count += 1
            else:
                ok_count = 0
            if ok_count >= 3 or not outer:
                break
            s += term
            for a in outer_a:
                outer *= a
            for b in outer_b:
                outer /= b
            m += 1
            outer = outer * x / m
            if m > maxterms:
                raise ctx.NoConvergence('maxterms exceeded in hyper2d')
    finally:
        ctx.prec = prec
    return +s