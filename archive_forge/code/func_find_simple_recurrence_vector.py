from sympy.concrete.products import (Product, product)
from sympy.core import Function, S
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import integrate
from sympy.polys.polyfuncs import rational_interpolate as rinterp
from sympy.polys.polytools import lcm
from sympy.simplify.radsimp import denom
from sympy.utilities import public
@public
def find_simple_recurrence_vector(l):
    """
    This function is used internally by other functions from the
    sympy.concrete.guess module. While most users may want to rather use the
    function find_simple_recurrence when looking for recurrence relations
    among rational numbers, the current function may still be useful when
    some post-processing has to be done.

    Explanation
    ===========

    The function returns a vector of length n when a recurrence relation of
    order n is detected in the sequence of rational numbers v.

    If the returned vector has a length 1, then the returned value is always
    the list [0], which means that no relation has been found.

    While the functions is intended to be used with rational numbers, it should
    work for other kinds of real numbers except for some cases involving
    quadratic numbers; for that reason it should be used with some caution when
    the argument is not a list of rational numbers.

    Examples
    ========

    >>> from sympy.concrete.guess import find_simple_recurrence_vector
    >>> from sympy import fibonacci
    >>> find_simple_recurrence_vector([fibonacci(k) for k in range(12)])
    [1, -1, -1]

    See Also
    ========

    See the function sympy.concrete.guess.find_simple_recurrence which is more
    user-friendly.

    """
    q1 = [0]
    q2 = [1]
    b, z = (0, len(l) >> 1)
    while len(q2) <= z:
        while l[b] == 0:
            b += 1
            if b == len(l):
                c = 1
                for x in q2:
                    c = lcm(c, denom(x))
                if q2[0] * c < 0:
                    c = -c
                for k in range(len(q2)):
                    q2[k] = int(q2[k] * c)
                return q2
        a = S.One / l[b]
        m = [a]
        for k in range(b + 1, len(l)):
            m.append(-sum((l[j + 1] * m[b - j - 1] for j in range(b, k))) * a)
        l, m = (m, [0] * max(len(q2), b + len(q1)))
        for k, q in enumerate(q2):
            m[k] = a * q
        for k, q in enumerate(q1):
            m[k + b] += q
        while m[-1] == 0:
            m.pop()
        q1, q2, b = (q2, m, 1)
    return [0]