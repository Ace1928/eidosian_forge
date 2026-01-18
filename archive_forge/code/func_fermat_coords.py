from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.numbers import igcdex, Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit
def fermat_coords(n: int) -> list[int] | None:
    """If n can be factored in terms of Fermat primes with
    multiplicity of each being 1, return those primes, else
    None
    """
    primes = []
    for p in [3, 5, 17, 257, 65537]:
        quotient, remainder = divmod(n, p)
        if remainder == 0:
            n = quotient
            primes.append(p)
            if n == 1:
                return primes
    return None