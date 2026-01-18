from sympy.core.function import diff
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)
from sympy.abc import a, q, z
def test_mathieus():
    assert isinstance(mathieus(a, q, z), mathieus)
    assert mathieus(a, 0, z) == sin(sqrt(a) * z)
    assert conjugate(mathieus(a, q, z)) == mathieus(conjugate(a), conjugate(q), conjugate(z))
    assert diff(mathieus(a, q, z), z) == mathieusprime(a, q, z)