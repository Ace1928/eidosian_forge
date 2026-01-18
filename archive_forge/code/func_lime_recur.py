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
def lime_recur(limits):
    x = self.dom.createElement('apply')
    x.appendChild(self.dom.createElement(self.mathml_tag(e)))
    bvar_elem = self.dom.createElement('bvar')
    bvar_elem.appendChild(self._print(limits[0][0]))
    x.appendChild(bvar_elem)
    if len(limits[0]) == 3:
        low_elem = self.dom.createElement('lowlimit')
        low_elem.appendChild(self._print(limits[0][1]))
        x.appendChild(low_elem)
        up_elem = self.dom.createElement('uplimit')
        up_elem.appendChild(self._print(limits[0][2]))
        x.appendChild(up_elem)
    if len(limits[0]) == 2:
        up_elem = self.dom.createElement('uplimit')
        up_elem.appendChild(self._print(limits[0][1]))
        x.appendChild(up_elem)
    if len(limits) == 1:
        x.appendChild(self._print(e.function))
    else:
        x.appendChild(lime_recur(limits[1:]))
    return x