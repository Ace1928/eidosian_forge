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
def as_coefficients_dict(self, *syms):
    """Return a dictionary mapping terms to their Rational coefficient.
        Since the dictionary is a defaultdict, inquiries about terms which
        were not present will return a coefficient of 0.

        If symbols ``syms`` are provided, any multiplicative terms
        independent of them will be considered a coefficient and a
        regular dictionary of syms-dependent generators as keys and
        their corresponding coefficients as values will be returned.

        Examples
        ========

        >>> from sympy.abc import a, x, y
        >>> (3*x + a*x + 4).as_coefficients_dict()
        {1: 4, x: 3, a*x: 1}
        >>> _[a]
        0
        >>> (3*a*x).as_coefficients_dict()
        {a*x: 3}
        >>> (3*a*x).as_coefficients_dict(x)
        {x: 3*a}
        >>> (3*a*x).as_coefficients_dict(y)
        {1: 3*a*x}

        """
    d = defaultdict(list)
    if not syms:
        for ai in Add.make_args(self):
            c, m = ai.as_coeff_Mul()
            d[m].append(c)
        for k, v in d.items():
            if len(v) == 1:
                d[k] = v[0]
            else:
                d[k] = Add(*v)
    else:
        ind, dep = self.as_independent(*syms, as_Add=True)
        for i in Add.make_args(dep):
            if i.is_Mul:
                c, x = i.as_coeff_mul(*syms)
                if c is S.One:
                    d[i].append(c)
                else:
                    d[i._new_rawargs(*x)].append(c)
            elif i:
                d[i].append(S.One)
        d = {k: Add(*d[k]) for k in d}
        if ind is not S.Zero:
            d.update({S.One: ind})
    di = defaultdict(int)
    di.update(d)
    return di