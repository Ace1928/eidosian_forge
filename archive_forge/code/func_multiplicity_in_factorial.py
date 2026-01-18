from collections import defaultdict
from functools import reduce
import random
import math
from sympy.core import sympify
from sympy.core.containers import Dict
from sympy.core.evalf import bitcount
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm, Rational, Integer
from sympy.core.power import integer_nthroot, Pow, integer_log
from sympy.core.singleton import S
from sympy.external.gmpy import SYMPY_INTS
from .primetest import isprime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor
def multiplicity_in_factorial(p, n):
    """return the largest integer ``m`` such that ``p**m`` divides ``n!``
    without calculating the factorial of ``n``.


    Examples
    ========

    >>> from sympy.ntheory import multiplicity_in_factorial
    >>> from sympy import factorial

    >>> multiplicity_in_factorial(2, 3)
    1

    An instructive use of this is to tell how many trailing zeros
    a given factorial has. For example, there are 6 in 25!:

    >>> factorial(25)
    15511210043330985984000000
    >>> multiplicity_in_factorial(10, 25)
    6

    For large factorials, it is much faster/feasible to use
    this function rather than computing the actual factorial:

    >>> multiplicity_in_factorial(factorial(25), 2**100)
    52818775009509558395695966887

    """
    p, n = (as_int(p), as_int(n))
    if p <= 0:
        raise ValueError('expecting positive integer got %s' % p)
    if n < 0:
        raise ValueError('expecting non-negative integer got %s' % n)
    factors = factorint(p)
    test = defaultdict(int)
    for k, v in factors.items():
        test[v] = max(k, test[v])
    keep = set(test.values())
    for k in list(factors.keys()):
        if k not in keep:
            factors.pop(k)
    mp = S.Infinity
    for i in factors:
        mi = (n - (sum(digits(n, i)) - i)) // (i - 1)
        mp = min(mp, mi // factors[i])
    return mp