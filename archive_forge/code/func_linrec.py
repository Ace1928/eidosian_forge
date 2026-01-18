from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def linrec(coeffs, init, n):
    """
    Evaluation of univariate linear recurrences of homogeneous type
    having coefficients independent of the recurrence variable.

    Parameters
    ==========

    coeffs : iterable
        Coefficients of the recurrence
    init : iterable
        Initial values of the recurrence
    n : Integer
        Point of evaluation for the recurrence

    Notes
    =====

    Let `y(n)` be the recurrence of given type, ``c`` be the sequence
    of coefficients, ``b`` be the sequence of initial/base values of the
    recurrence and ``k`` (equal to ``len(c)``) be the order of recurrence.
    Then,

    .. math :: y(n) = \\begin{cases} b_n & 0 \\le n < k \\\\
        c_0 y(n-1) + c_1 y(n-2) + \\cdots + c_{k-1} y(n-k) & n \\ge k
        \\end{cases}

    Let `x_0, x_1, \\ldots, x_n` be a sequence and consider the transformation
    that maps each polynomial `f(x)` to `T(f(x))` where each power `x^i` is
    replaced by the corresponding value `x_i`. The sequence is then a solution
    of the recurrence if and only if `T(x^i p(x)) = 0` for each `i \\ge 0` where
    `p(x) = x^k - c_0 x^(k-1) - \\cdots - c_{k-1}` is the characteristic
    polynomial.

    Then `T(f(x)p(x)) = 0` for each polynomial `f(x)` (as it is a linear
    combination of powers `x^i`). Now, if `x^n` is congruent to
    `g(x) = a_0 x^0 + a_1 x^1 + \\cdots + a_{k-1} x^{k-1}` modulo `p(x)`, then
    `T(x^n) = x_n` is equal to
    `T(g(x)) = a_0 x_0 + a_1 x_1 + \\cdots + a_{k-1} x_{k-1}`.

    Computation of `x^n`,
    given `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \\cdots + c_{k-1}`
    is performed using exponentiation by squaring (refer to [1_]) with
    an additional reduction step performed to retain only first `k` powers
    of `x` in the representation of `x^n`.

    Examples
    ========

    >>> from sympy.discrete.recurrences import linrec
    >>> from sympy.abc import x, y, z

    >>> linrec(coeffs=[1, 1], init=[0, 1], n=10)
    55

    >>> linrec(coeffs=[1, 1], init=[x, y], n=10)
    34*x + 55*y

    >>> linrec(coeffs=[x, y], init=[0, 1], n=5)
    x**2*y + x*(x**3 + 2*x*y) + y**2

    >>> linrec(coeffs=[1, 2, 3, 0, 0, 4], init=[x, y, z], n=16)
    13576*x + 5676*y + 2356*z

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    .. [2] https://en.wikipedia.org/w/index.php?title=Modular_exponentiation&section=6#Matrices

    See Also
    ========

    sympy.polys.agca.extensions.ExtensionElement.__pow__

    """
    if not coeffs:
        return S.Zero
    if not iterable(coeffs):
        raise TypeError('Expected a sequence of coefficients for the recurrence')
    if not iterable(init):
        raise TypeError('Expected a sequence of values for the initialization of the recurrence')
    n = as_int(n)
    if n < 0:
        raise ValueError('Point of evaluation of recurrence must be a non-negative integer')
    c = [sympify(arg) for arg in coeffs]
    b = [sympify(arg) for arg in init]
    k = len(c)
    if len(b) > k:
        raise TypeError('Count of initial values should not exceed the order of the recurrence')
    else:
        b += [S.Zero] * (k - len(b))
    if n < k:
        return b[n]
    terms = [u * v for u, v in zip(linrec_coeffs(c, n), b)]
    return sum(terms[:-1], terms[-1])