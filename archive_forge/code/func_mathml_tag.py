from __future__ import annotations
from typing import Any
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import \
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.printer import Printer, print_function
from mpmath.libmp import prec_to_dps, repr_dps, to_str as mlib_to_str
def mathml_tag(self, e):
    """Returns the MathML tag for an expression."""
    translate = {'Number': 'mn', 'Limit': '&#x2192;', 'Derivative': '&dd;', 'int': 'mn', 'Symbol': 'mi', 'Integral': '&int;', 'Sum': '&#x2211;', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot', 'asin': 'arcsin', 'asinh': 'arcsinh', 'acos': 'arccos', 'acosh': 'arccosh', 'atan': 'arctan', 'atanh': 'arctanh', 'acot': 'arccot', 'atan2': 'arctan', 'Equality': '=', 'Unequality': '&#x2260;', 'GreaterThan': '&#x2265;', 'LessThan': '&#x2264;', 'StrictGreaterThan': '>', 'StrictLessThan': '<', 'lerchphi': '&#x3A6;', 'zeta': '&#x3B6;', 'dirichlet_eta': '&#x3B7;', 'elliptic_k': '&#x39A;', 'lowergamma': '&#x3B3;', 'uppergamma': '&#x393;', 'gamma': '&#x393;', 'totient': '&#x3D5;', 'reduced_totient': '&#x3BB;', 'primenu': '&#x3BD;', 'primeomega': '&#x3A9;', 'fresnels': 'S', 'fresnelc': 'C', 'LambertW': 'W', 'Heaviside': '&#x398;', 'BooleanTrue': 'True', 'BooleanFalse': 'False', 'NoneType': 'None', 'mathieus': 'S', 'mathieuc': 'C', 'mathieusprime': 'S&#x2032;', 'mathieucprime': 'C&#x2032;'}

    def mul_symbol_selection():
        if self._settings['mul_symbol'] is None or self._settings['mul_symbol'] == 'None':
            return '&InvisibleTimes;'
        elif self._settings['mul_symbol'] == 'times':
            return '&#xD7;'
        elif self._settings['mul_symbol'] == 'dot':
            return '&#xB7;'
        elif self._settings['mul_symbol'] == 'ldot':
            return '&#x2024;'
        elif not isinstance(self._settings['mul_symbol'], str):
            raise TypeError
        else:
            return self._settings['mul_symbol']
    for cls in e.__class__.__mro__:
        n = cls.__name__
        if n in translate:
            return translate[n]
    if e.__class__.__name__ == 'Mul':
        return mul_symbol_selection()
    n = e.__class__.__name__
    return n.lower()