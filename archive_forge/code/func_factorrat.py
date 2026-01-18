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
def factorrat(rat, limit=None, use_trial=True, use_rho=True, use_pm1=True, verbose=False, visual=None, multiple=False):
    """
    Given a Rational ``r``, ``factorrat(r)`` returns a dict containing
    the prime factors of ``r`` as keys and their respective multiplicities
    as values. For example:

    >>> from sympy import factorrat, S
    >>> factorrat(S(8)/9)    # 8/9 = (2**3) * (3**-2)
    {2: 3, 3: -2}
    >>> factorrat(S(-1)/987)    # -1/789 = -1 * (3**-1) * (7**-1) * (47**-1)
    {-1: 1, 3: -1, 7: -1, 47: -1}

    Please see the docstring for ``factorint`` for detailed explanations
    and examples of the following keywords:

        - ``limit``: Integer limit up to which trial division is done
        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method
        - ``verbose``: Toggle detailed printing of progress
        - ``multiple``: Toggle returning a list of factors or dict
        - ``visual``: Toggle product form of output
    """
    if multiple:
        fac = factorrat(rat, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One / p] * -fac[p] for p, _ in sorted(fac.items(), key=lambda elem: elem[0] if elem[1] > 0 else 1 / elem[0])), [])
        return factorlist
    f = factorint(rat.p, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose).copy()
    f = defaultdict(int, f)
    for p, e in factorint(rat.q, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose).items():
        f[p] += -e
    if len(f) > 1 and 1 in f:
        del f[1]
    if not visual:
        return dict(f)
    else:
        if -1 in f:
            f.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False) for i in sorted(f.items())])
        return Mul(*args, evaluate=False)