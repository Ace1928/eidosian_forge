from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def parse_hints(hints):
    """Split hints into (n, funcs, iterables, gens)."""
    n = 1
    funcs, iterables, gens = ([], [], [])
    for e in hints:
        if isinstance(e, (SYMPY_INTS, Integer)):
            n = e
        elif isinstance(e, FunctionClass):
            funcs.append(e)
        elif iterable(e):
            iterables.append((e[0], e[1:]))
            gens.extend(parallel_poly_from_expr([e[0](x) for x in e[1:]] + [e[0](Add(*e[1:]))])[1].gens)
        else:
            gens.append(e)
    return (n, funcs, iterables, gens)