from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable
from functools import reduce
import re
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .cache import cacheit
from .sorting import default_sort_key
from .kind import NumberKind
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int, func_name, filldedent
from sympy.utilities.iterables import has_variety, sift
from mpmath.libmp import mpf_log, prec_to_dps
from mpmath.libmp.libintmath import giant_steps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Function, _derivative_dispatch
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Float, Integer, Rational, _illegal
def as_coeff_exponent(self, x) -> tuple[Expr, Expr]:
    """ ``c*x**e -> c,e`` where x can be any symbolic expression.
        """
    from sympy.simplify.radsimp import collect
    s = collect(self, x)
    c, p = s.as_coeff_mul(x)
    if len(p) == 1:
        b, e = p[0].as_base_exp()
        if b == x:
            return (c, e)
    return (s, S.Zero)