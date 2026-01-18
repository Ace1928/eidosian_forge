from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_issue_14041():
    import sympy.physics.mechanics as me
    A_frame = me.ReferenceFrame('A')
    thetad, phid = me.dynamicsymbols('theta, phi', 1)
    L = symbols('L')
    assert vlatex(L * (phid + thetad) ** 2 * A_frame.x) == 'L \\left(\\dot{\\phi} + \\dot{\\theta}\\right)^{2}\\mathbf{\\hat{a}_x}'
    assert vlatex((phid + thetad) ** 2 * A_frame.x) == '\\left(\\dot{\\phi} + \\dot{\\theta}\\right)^{2}\\mathbf{\\hat{a}_x}'
    assert vlatex((phid * thetad) ** a * A_frame.x) == '\\left(\\dot{\\phi} \\dot{\\theta}\\right)^{a}\\mathbf{\\hat{a}_x}'