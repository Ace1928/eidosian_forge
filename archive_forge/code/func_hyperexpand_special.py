from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def hyperexpand_special(ap, bq, z):
    """
    Try to find a closed-form expression for hyper(ap, bq, z), where ``z``
    is supposed to be a "special" value, e.g. 1.

    This function tries various of the classical summation formulae
    (Gauss, Saalschuetz, etc).
    """
    p, q = (len(ap), len(bq))
    z_ = z
    z = unpolarify(z)
    if z == 0:
        return S.One
    from sympy.simplify.simplify import simplify
    if p == 2 and q == 1:
        a, b, c = ap + bq
        if z == 1:
            return gamma(c - a - b) * gamma(c) / gamma(c - a) / gamma(c - b)
        if z == -1 and simplify(b - a + c) == 1:
            b, a = (a, b)
        if z == -1 and simplify(a - b + c) == 1:
            if b.is_integer and b.is_negative:
                return 2 * cos(pi * b / 2) * gamma(-b) * gamma(b - a + 1) / gamma(-b / 2) / gamma(b / 2 - a + 1)
            else:
                return gamma(b / 2 + 1) * gamma(b - a + 1) / gamma(b + 1) / gamma(b / 2 - a + 1)
    return hyper(ap, bq, z_)