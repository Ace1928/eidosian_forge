from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
def test_dyadic_pretty_print():
    expected = ' 2\na  n_x|n_y + b n_y|n_y + c*sin(alpha) n_z|n_y'
    uexpected = ' 2\na  n_x⊗n_y + b n_y⊗n_y + c⋅sin(α) n_z⊗n_y'
    assert ascii_vpretty(y) == expected
    assert unicode_vpretty(y) == uexpected
    expected = 'alpha n_x|n_x + sin(omega) n_y|n_z + alpha*beta n_z|n_x'
    uexpected = 'α n_x⊗n_x + sin(ω) n_y⊗n_z + α⋅β n_z⊗n_x'
    assert ascii_vpretty(x) == expected
    assert unicode_vpretty(x) == uexpected
    assert ascii_vpretty(Dyadic([])) == '0'
    assert unicode_vpretty(Dyadic([])) == '0'
    assert ascii_vpretty(xx) == '- n_x|n_y - n_x|n_z'
    assert unicode_vpretty(xx) == '- n_x⊗n_y - n_x⊗n_z'
    assert ascii_vpretty(xx2) == 'n_x|n_y + n_x|n_z'
    assert unicode_vpretty(xx2) == 'n_x⊗n_y + n_x⊗n_z'