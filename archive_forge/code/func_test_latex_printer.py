from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_latex_printer():
    r = Function('r')('t')
    assert VectorLatexPrinter().doprint(r ** 2) == 'r^{2}'
    r2 = Function('r^2')('t')
    assert VectorLatexPrinter().doprint(r2.diff()) == '\\dot{r^{2}}'
    ra = Function('r__a')('t')
    assert VectorLatexPrinter().doprint(ra.diff().diff()) == '\\ddot{r^{a}}'