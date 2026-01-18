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
def as_terms(self):
    """Transform an expression to a list of terms. """
    from .exprtools import decompose_power
    gens, terms = (set(), [])
    for term in Add.make_args(self):
        coeff, _term = term.as_coeff_Mul()
        coeff = complex(coeff)
        cpart, ncpart = ({}, [])
        if _term is not S.One:
            for factor in Mul.make_args(_term):
                if factor.is_number:
                    try:
                        coeff *= complex(factor)
                    except (TypeError, ValueError):
                        pass
                    else:
                        continue
                if factor.is_commutative:
                    base, exp = decompose_power(factor)
                    cpart[base] = exp
                    gens.add(base)
                else:
                    ncpart.append(factor)
        coeff = (coeff.real, coeff.imag)
        ncpart = tuple(ncpart)
        terms.append((term, (coeff, cpart, ncpart)))
    gens = sorted(gens, key=default_sort_key)
    k, indices = (len(gens), {})
    for i, g in enumerate(gens):
        indices[g] = i
    result = []
    for term, (coeff, cpart, ncpart) in terms:
        monom = [0] * k
        for base, exp in cpart.items():
            monom[indices[base]] = exp
        result.append((term, (coeff, tuple(monom), ncpart)))
    return (result, gens)