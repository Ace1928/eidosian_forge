from __future__ import annotations
from typing import Tuple as tTuple, Optional, Union as tUnion, Callable, List, Dict as tDict, Type, TYPE_CHECKING, \
import math
import mpmath.libmp as libmp
from mpmath import (
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int
def add_terms(terms: list, prec: int, target_prec: int) -> tTuple[tUnion[MPF_TUP, SCALED_ZERO_TUP, None], Optional[int]]:
    """
    Helper for evalf_add. Adds a list of (mpfval, accuracy) terms.

    Returns
    =======

    - None, None if there are no non-zero terms;
    - terms[0] if there is only 1 term;
    - scaled_zero if the sum of the terms produces a zero by cancellation
      e.g. mpfs representing 1 and -1 would produce a scaled zero which need
      special handling since they are not actually zero and they are purposely
      malformed to ensure that they cannot be used in anything but accuracy
      calculations;
    - a tuple that is scaled to target_prec that corresponds to the
      sum of the terms.

    The returned mpf tuple will be normalized to target_prec; the input
    prec is used to define the working precision.

    XXX explain why this is needed and why one cannot just loop using mpf_add
    """
    terms = [t for t in terms if not iszero(t[0])]
    if not terms:
        return (None, None)
    elif len(terms) == 1:
        return terms[0]
    special = []
    from .numbers import Float
    for t in terms:
        arg = Float._new(t[0], 1)
        if arg is S.NaN or arg.is_infinite:
            special.append(arg)
    if special:
        from .add import Add
        rv = evalf(Add(*special), prec + 4, {})
        return (rv[0], rv[2])
    working_prec = 2 * prec
    sum_man, sum_exp = (0, 0)
    absolute_err: List[int] = []
    for x, accuracy in terms:
        sign, man, exp, bc = x
        if sign:
            man = -man
        absolute_err.append(bc + exp - accuracy)
        delta = exp - sum_exp
        if exp >= sum_exp:
            if delta > working_prec and (not sum_man or delta - bitcount(abs(sum_man)) > working_prec):
                sum_man = man
                sum_exp = exp
            else:
                sum_man += man << delta
        else:
            delta = -delta
            if delta - bc > working_prec:
                if not sum_man:
                    sum_man, sum_exp = (man, exp)
            else:
                sum_man = (sum_man << delta) + man
                sum_exp = exp
    absolute_error = max(absolute_err)
    if not sum_man:
        return scaled_zero(absolute_error)
    if sum_man < 0:
        sum_sign = 1
        sum_man = -sum_man
    else:
        sum_sign = 0
    sum_bc = bitcount(sum_man)
    sum_accuracy = sum_exp + sum_bc - absolute_error
    r = (normalize(sum_sign, sum_man, sum_exp, sum_bc, target_prec, rnd), sum_accuracy)
    return r