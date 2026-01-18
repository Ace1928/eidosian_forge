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
def hyperexpand(f, allow_hyper=False, rewrite='default', place=None):
    """
    Expand hypergeometric functions. If allow_hyper is True, allow partial
    simplification (that is a result different from input,
    but still containing hypergeometric functions).

    If a G-function has expansions both at zero and at infinity,
    ``place`` can be set to ``0`` or ``zoo`` to indicate the
    preferred choice.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import hyperexpand
    >>> from sympy.functions import hyper
    >>> from sympy.abc import z
    >>> hyperexpand(hyper([], [], z))
    exp(z)

    Non-hyperegeometric parts of the expression and hypergeometric expressions
    that are not recognised are left unchanged:

    >>> hyperexpand(1 + hyper([1, 1, 1], [], z))
    hyper((1, 1, 1), (), z) + 1
    """
    f = sympify(f)

    def do_replace(ap, bq, z):
        r = _hyperexpand(Hyper_Function(ap, bq), z, rewrite=rewrite)
        if r is None:
            return hyper(ap, bq, z)
        else:
            return r

    def do_meijer(ap, bq, z):
        r = _meijergexpand(G_Function(ap[0], ap[1], bq[0], bq[1]), z, allow_hyper, rewrite=rewrite, place=place)
        if not r.has(nan, zoo, oo, -oo):
            return r
    return f.replace(hyper, do_replace).replace(meijerg, do_meijer)