from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import itertools
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse
from sympy.tensor.array import NDimArray
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.utilities.iterables import has_variety, sift
import re
def convert_args(args) -> str:
    _tex = last_term_tex = ''
    for i, term in enumerate(args):
        term_tex = self._print(term)
        if not (hasattr(term, '_scale_factor') or hasattr(term, 'is_physical_constant')):
            if self._needs_mul_brackets(term, first=i == 0, last=i == len(args) - 1):
                term_tex = '\\left(%s\\right)' % term_tex
            if _between_two_numbers_p[0].search(last_term_tex) and _between_two_numbers_p[1].match(str(term)):
                _tex += numbersep
            elif _tex:
                _tex += separator
        elif _tex:
            _tex += separator
        _tex += term_tex
        last_term_tex = term_tex
    return _tex