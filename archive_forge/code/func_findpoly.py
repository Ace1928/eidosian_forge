from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed
def findpoly(ctx, x, n=1, **kwargs):
    """
    ``findpoly(x, n)`` returns the coefficients of an integer
    polynomial `P` of degree at most `n` such that `P(x) \\approx 0`.
    If no polynomial having `x` as a root can be found,
    :func:`~mpmath.findpoly` returns ``None``.

    :func:`~mpmath.findpoly` works by successively calling :func:`~mpmath.pslq` with
    the vectors `[1, x]`, `[1, x, x^2]`, `[1, x, x^2, x^3]`, ...,
    `[1, x, x^2, .., x^n]` as input. Keyword arguments given to
    :func:`~mpmath.findpoly` are forwarded verbatim to :func:`~mpmath.pslq`. In
    particular, you can specify a tolerance for `P(x)` with ``tol``
    and a maximum permitted coefficient size with ``maxcoeff``.

    For large values of `n`, it is recommended to run :func:`~mpmath.findpoly`
    at high precision; preferably 50 digits or more.

    **Examples**

    By default (degree `n = 1`), :func:`~mpmath.findpoly` simply finds a linear
    polynomial with a rational root::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> findpoly(0.7)
        [-10, 7]

    The generated coefficient list is valid input to ``polyval`` and
    ``polyroots``::

        >>> nprint(polyval(findpoly(phi, 2), phi), 1)
        -2.0e-16
        >>> for r in polyroots(findpoly(phi, 2)):
        ...     print(r)
        ...
        -0.618033988749895
        1.61803398874989

    Numbers of the form `m + n \\sqrt p` for integers `(m, n, p)` are
    solutions to quadratic equations. As we find here, `1+\\sqrt 2`
    is a root of the polynomial `x^2 - 2x - 1`::

        >>> findpoly(1+sqrt(2), 2)
        [1, -2, -1]
        >>> findroot(lambda x: x**2 - 2*x - 1, 1)
        2.4142135623731

    Despite only containing square roots, the following number results
    in a polynomial of degree 4::

        >>> findpoly(sqrt(2)+sqrt(3), 4)
        [1, 0, -10, 0, 1]

    In fact, `x^4 - 10x^2 + 1` is the *minimal polynomial* of
    `r = \\sqrt 2 + \\sqrt 3`, meaning that a rational polynomial of
    lower degree having `r` as a root does not exist. Given sufficient
    precision, :func:`~mpmath.findpoly` will usually find the correct
    minimal polynomial of a given algebraic number.

    **Non-algebraic numbers**

    If :func:`~mpmath.findpoly` fails to find a polynomial with given
    coefficient size and tolerance constraints, that means no such
    polynomial exists.

    We can verify that `\\pi` is not an algebraic number of degree 3 with
    coefficients less than 1000::

        >>> mp.dps = 15
        >>> findpoly(pi, 3)
        >>>

    It is always possible to find an algebraic approximation of a number
    using one (or several) of the following methods:

        1. Increasing the permitted degree
        2. Allowing larger coefficients
        3. Reducing the tolerance

    One example of each method is shown below::

        >>> mp.dps = 15
        >>> findpoly(pi, 4)
        [95, -545, 863, -183, -298]
        >>> findpoly(pi, 3, maxcoeff=10000)
        [836, -1734, -2658, -457]
        >>> findpoly(pi, 3, tol=1e-7)
        [-4, 22, -29, -2]

    It is unknown whether Euler's constant is transcendental (or even
    irrational). We can use :func:`~mpmath.findpoly` to check that if is
    an algebraic number, its minimal polynomial must have degree
    at least 7 and a coefficient of magnitude at least 1000000::

        >>> mp.dps = 200
        >>> findpoly(euler, 6, maxcoeff=10**6, tol=1e-100, maxsteps=1000)
        >>>

    Note that the high precision and strict tolerance is necessary
    for such high-degree runs, since otherwise unwanted low-accuracy
    approximations will be detected. It may also be necessary to set
    maxsteps high to prevent a premature exit (before the coefficient
    bound has been reached). Running with ``verbose=True`` to get an
    idea what is happening can be useful.
    """
    x = ctx.mpf(x)
    if n < 1:
        raise ValueError('n cannot be less than 1')
    if x == 0:
        return [1, 0]
    xs = [ctx.mpf(1)]
    for i in range(1, n + 1):
        xs.append(x ** i)
        a = ctx.pslq(xs, **kwargs)
        if a is not None:
            return a[::-1]