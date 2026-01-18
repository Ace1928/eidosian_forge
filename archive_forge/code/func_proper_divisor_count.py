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
def proper_divisor_count(n, modulus=1):
    """
    Return the number of proper divisors of ``n``.

    Examples
    ========

    >>> from sympy import proper_divisor_count
    >>> proper_divisor_count(6)
    3
    >>> proper_divisor_count(6, modulus=2)
    1

    See Also
    ========

    divisors, proper_divisors, divisor_count

    """
    return divisor_count(n, modulus=modulus, proper=True)