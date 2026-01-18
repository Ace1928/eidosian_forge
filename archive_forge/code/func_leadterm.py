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
def leadterm(self, x, logx=None, cdir=0):
    """
        Returns the leading term a*x**b as a tuple (a, b).

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1+x+x**2).leadterm(x)
        (1, 0)
        >>> (1/x**2+x+x**2).leadterm(x)
        (1, -2)

        """
    from .symbol import Dummy
    from sympy.functions.elementary.exponential import log
    l = self.as_leading_term(x, logx=logx, cdir=cdir)
    d = Dummy('logx')
    if l.has(log(x)):
        l = l.subs(log(x), d)
    c, e = l.as_coeff_exponent(x)
    if x in c.free_symbols:
        raise ValueError(filldedent('\n                cannot compute leadterm(%s, %s). The coefficient\n                should have been free of %s but got %s' % (self, x, x, c)))
    c = c.subs(d, log(x))
    return (c, e)