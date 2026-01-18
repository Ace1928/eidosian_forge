from .functions import defun, defun_wrapped
@defun
def qhyper(ctx, a_s, b_s, q, z, **kwargs):
    """
    Evaluates the basic hypergeometric series or hypergeometric q-series

    .. math ::

        \\,_r\\phi_s \\left[\\begin{matrix}
            a_1 & a_2 & \\ldots & a_r \\\\
            b_1 & b_2 & \\ldots & b_s
        \\end{matrix} ; q,z \\right] =
        \\sum_{n=0}^\\infty
        \\frac{(a_1;q)_n, \\ldots, (a_r;q)_n}
             {(b_1;q)_n, \\ldots, (b_s;q)_n}
        \\left((-1)^n q^{n\\choose 2}\\right)^{1+s-r}
        \\frac{z^n}{(q;q)_n}

    where `(a;q)_n` denotes the q-Pochhammer symbol (see :func:`~mpmath.qp`).

    **Examples**

    Evaluation works for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qhyper([0.5], [2.25], 0.25, 4)
        -0.1975849091263356009534385
        >>> qhyper([0.5], [2.25], 0.25-0.25j, 4)
        (2.806330244925716649839237 + 3.568997623337943121769938j)
        >>> qhyper([1+j], [2,3+0.5j], 0.25, 3+4j)
        (9.112885171773400017270226 - 1.272756997166375050700388j)

    Comparing with a summation of the defining series, using
    :func:`~mpmath.nsum`::

        >>> b, q, z = 3, 0.25, 0.5
        >>> qhyper([], [b], q, z)
        0.6221136748254495583228324
        >>> nsum(lambda n: z**n / qp(q,q,n)/qp(b,q,n) * q**(n*(n-1)), [0,inf])
        0.6221136748254495583228324

    """
    a_s = [ctx.convert(a) for a in a_s]
    b_s = [ctx.convert(b) for b in b_s]
    q = ctx.convert(q)
    z = ctx.convert(z)
    r = len(a_s)
    s = len(b_s)
    d = 1 + s - r
    maxterms = kwargs.get('maxterms', 50 * ctx.prec)

    def terms():
        t = ctx.one
        yield t
        qk = 1
        k = 0
        x = 1
        while 1:
            for a in a_s:
                p = 1 - a * qk
                t *= p
            for b in b_s:
                p = 1 - b * qk
                if not p:
                    raise ValueError
                t /= p
            t *= z
            x *= (-1) ** d * qk ** d
            qk *= q
            t /= 1 - qk
            k += 1
            yield (t * x)
            if k > maxterms:
                raise ctx.NoConvergence
    return ctx.sum_accurately(terms)