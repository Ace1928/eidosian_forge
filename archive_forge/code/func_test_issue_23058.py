from sympy.core.function import Function
from sympy.integrals.integrals import Integral
from sympy.printing.latex import latex
from sympy.printing.pretty import pretty as xpretty
from sympy.vector import CoordSys3D, Del, Vector, express
from sympy.abc import a, b, c
from sympy.testing.pytest import XFAIL
def test_issue_23058():
    from sympy import symbols, sin, cos, pi, UnevaluatedExpr
    delop = Del()
    CC_ = CoordSys3D('C')
    y = CC_.y
    xhat = CC_.i
    t = symbols('t')
    ten = symbols('10', positive=True)
    eps, mu = (4 * pi * ten ** (-11), ten ** (-5))
    Bx = 2 * ten ** (-4) * cos(ten ** 5 * t) * sin(ten ** (-3) * y)
    vecB = Bx * xhat
    vecE = 1 / eps * Integral(delop.cross(vecB / mu).doit(), t)
    vecE = vecE.doit()
    vecB_str = '⎛     ⎛y_C⎞    ⎛  5  ⎞⎞    \n⎜2⋅sin⎜───⎟⋅cos⎝10 ⋅t⎠⎟ i_C\n⎜     ⎜  3⎟           ⎟    \n⎜     ⎝10 ⎠           ⎟    \n⎜─────────────────────⎟    \n⎜           4         ⎟    \n⎝         10          ⎠    '
    vecE_str = '⎛   4    ⎛  5  ⎞    ⎛y_C⎞ ⎞    \n⎜-10 ⋅sin⎝10 ⋅t⎠⋅cos⎜───⎟ ⎟ k_C\n⎜                   ⎜  3⎟ ⎟    \n⎜                   ⎝10 ⎠ ⎟    \n⎜─────────────────────────⎟    \n⎝           2⋅π           ⎠    '
    assert upretty(vecB) == vecB_str
    assert upretty(vecE) == vecE_str
    ten = UnevaluatedExpr(10)
    eps, mu = (4 * pi * ten ** (-11), ten ** (-5))
    Bx = 2 * ten ** (-4) * cos(ten ** 5 * t) * sin(ten ** (-3) * y)
    vecB = Bx * xhat
    vecB_str = '⎛    -4    ⎛    5⎞    ⎛      -3⎞⎞     \n⎝2⋅10  ⋅cos⎝t⋅10 ⎠⋅sin⎝y_C⋅10  ⎠⎠ i_C '
    assert upretty(vecB) == vecB_str