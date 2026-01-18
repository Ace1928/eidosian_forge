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
def extract_additively(self, c):
    """Return self - c if it's possible to subtract c from self and
        make all matching coefficients move towards zero, else return None.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = 2*x + 3
        >>> e.extract_additively(x + 1)
        x + 2
        >>> e.extract_additively(3*x)
        >>> e.extract_additively(4)
        >>> (y*(x + 1)).extract_additively(x + 1)
        >>> ((x + 1)*(x + 2*y + 1) + 3).extract_additively(x + 1)
        (x + 1)*(x + 2*y) + 3

        See Also
        ========
        extract_multiplicatively
        coeff
        as_coefficient

        """
    c = sympify(c)
    if self is S.NaN:
        return None
    if c.is_zero:
        return self
    elif c == self:
        return S.Zero
    elif self == S.Zero:
        return None
    if self.is_Number:
        if not c.is_Number:
            return None
        co = self
        diff = co - c
        if co > 0 and diff >= 0 and (diff < co) or (co < 0 and diff <= 0 and (diff > co)):
            return diff
        return None
    if c.is_Number:
        co, t = self.as_coeff_Add()
        xa = co.extract_additively(c)
        if xa is None:
            return None
        return xa + t
    if c.is_Add and c.args[0].is_Number:
        co = self.coeff(c)
        xa0 = (co.extract_additively(1) or 0) * c
        if xa0:
            diff = self - co * c
            return xa0 + (diff.extract_additively(c) or diff) or None
        h, t = c.as_coeff_Add()
        sh, st = self.as_coeff_Add()
        xa = sh.extract_additively(h)
        if xa is None:
            return None
        xa2 = st.extract_additively(t)
        if xa2 is None:
            return None
        return xa + xa2
    co, diff = _corem(self, c)
    xa0 = (co.extract_additively(1) or 0) * c
    if xa0:
        return xa0 + (diff.extract_additively(c) or diff) or None
    coeffs = []
    for a in Add.make_args(c):
        ac, at = a.as_coeff_Mul()
        co = self.coeff(at)
        if not co:
            return None
        coc, cot = co.as_coeff_Add()
        xa = coc.extract_additively(ac)
        if xa is None:
            return None
        self -= co * at
        coeffs.append((cot + xa) * at)
    coeffs.append(self)
    return Add(*coeffs)