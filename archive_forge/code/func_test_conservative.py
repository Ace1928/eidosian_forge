from sympy.core.function import Derivative
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.simplify import simplify
from sympy.core.symbol import symbols
from sympy.core import S
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.vector import Dot
from sympy.vector.operators import curl, divergence, gradient, Gradient, Divergence, Cross
from sympy.vector.deloperator import Del
from sympy.vector.functions import (is_conservative, is_solenoidal,
from sympy.testing.pytest import raises
def test_conservative():
    assert is_conservative(Vector.zero) is True
    assert is_conservative(i) is True
    assert is_conservative(2 * i + 3 * j + 4 * k) is True
    assert is_conservative(y * z * i + x * z * j + x * y * k) is True
    assert is_conservative(x * j) is False
    assert is_conservative(grad_field) is True
    assert is_conservative(curl_field) is False
    assert is_conservative(4 * x * y * z * i + 2 * x ** 2 * z * j) is False
    assert is_conservative(z * P.i + P.x * k) is True