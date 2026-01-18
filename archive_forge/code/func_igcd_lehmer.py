from __future__ import annotations
import numbers
import decimal
import fractions
import math
import re as regex
import sys
from functools import lru_cache
from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .logic import fuzzy_not
from .kind import NumberKind
from sympy.external.gmpy import SYMPY_INTS, HAS_GMPY, gmpy
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
from sympy.utilities.misc import as_int, debug, filldedent
from .parameters import global_parameters
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
def igcd_lehmer(a, b):
    """Computes greatest common divisor of two integers.

    Explanation
    ===========

    Euclid's algorithm for the computation of the greatest
    common divisor ``gcd(a, b)``  of two (positive) integers
    $a$ and $b$ is based on the division identity
       $$ a = q \\times b + r$$,
    where the quotient  $q$  and the remainder  $r$  are integers
    and  $0 \\le r < b$. Then each common divisor of  $a$  and  $b$
    divides  $r$, and it follows that  ``gcd(a, b) == gcd(b, r)``.
    The algorithm works by constructing the sequence
    r0, r1, r2, ..., where  r0 = a, r1 = b,  and each  rn
    is the remainder from the division of the two preceding
    elements.

    In Python, ``q = a // b``  and  ``r = a % b``  are obtained by the
    floor division and the remainder operations, respectively.
    These are the most expensive arithmetic operations, especially
    for large  a  and  b.

    Lehmer's algorithm [1]_ is based on the observation that the quotients
    ``qn = r(n-1) // rn``  are in general small integers even
    when  a  and  b  are very large. Hence the quotients can be
    usually determined from a relatively small number of most
    significant bits.

    The efficiency of the algorithm is further enhanced by not
    computing each long remainder in Euclid's sequence. The remainders
    are linear combinations of  a  and  b  with integer coefficients
    derived from the quotients. The coefficients can be computed
    as far as the quotients can be determined from the chosen
    most significant parts of  a  and  b. Only then a new pair of
    consecutive remainders is computed and the algorithm starts
    anew with this pair.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lehmer%27s_GCD_algorithm

    """
    a, b = (abs(as_int(a)), abs(as_int(b)))
    if a < b:
        a, b = (b, a)
    nbits = 2 * sys.int_info.bits_per_digit
    while a.bit_length() > nbits and b != 0:
        n = a.bit_length() - nbits
        x, y = (int(a >> n), int(b >> n))
        A, B, C, D = (1, 0, 0, 1)
        while True:
            if y + C <= 0:
                break
            q = (x + A) // (y + C)
            x_qy, B_qD = (x - q * y, B - q * D)
            if x_qy + B_qD < 0:
                break
            x, y = (y, x_qy)
            A, B, C, D = (C, D, A - q * C, B_qD)
            if y + D <= 0:
                break
            q = (x + B) // (y + D)
            x_qy, A_qC = (x - q * y, A - q * C)
            if x_qy + A_qC < 0:
                break
            x, y = (y, x_qy)
            A, B, C, D = (C, D, A_qC, B - q * D)
        if B == 0:
            a, b = (b, a % b)
            continue
        a, b = (A * a + B * b, C * a + D * b)
    while b:
        a, b = (b, a % b)
    return a