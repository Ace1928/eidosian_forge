from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
def cornacchia(a, b, m):
    """
    Solves `ax^2 + by^2 = m` where `\\gcd(a, b) = 1 = gcd(a, m)` and `a, b > 0`.

    Explanation
    ===========

    Uses the algorithm due to Cornacchia. The method only finds primitive
    solutions, i.e. ones with `\\gcd(x, y) = 1`. So this method cannot be used to
    find the solutions of `x^2 + y^2 = 20` since the only solution to former is
    `(x, y) = (4, 2)` and it is not primitive. When `a = b`, only the
    solutions with `x \\leq y` are found. For more details, see the References.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import cornacchia
    >>> cornacchia(2, 3, 35) # equation 2x**2 + 3y**2 = 35
    {(2, 3), (4, 1)}
    >>> cornacchia(1, 1, 25) # equation x**2 + y**2 = 25
    {(4, 3)}

    References
    ===========

    .. [1] A. Nitaj, "L'algorithme de Cornacchia"
    .. [2] Solving the diophantine equation ax**2 + by**2 = m by Cornacchia's
        method, [online], Available:
        http://www.numbertheory.org/php/cornacchia.html

    See Also
    ========

    sympy.utilities.iterables.signed_permutations
    """
    sols = set()
    a1 = igcdex(a, m)[0]
    v = sqrt_mod(-b * a1, m, all_roots=True)
    if not v:
        return None
    for t in v:
        if t < m // 2:
            continue
        u, r = (t, m)
        while True:
            u, r = (r, u % r)
            if a * r ** 2 < m:
                break
        m1 = m - a * r ** 2
        if m1 % b == 0:
            m1 = m1 // b
            s, _exact = integer_nthroot(m1, 2)
            if _exact:
                if a == b and r < s:
                    r, s = (s, r)
                sols.add((int(r), int(s)))
    return sols