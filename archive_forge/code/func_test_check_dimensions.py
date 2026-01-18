from sympy.core.containers import Tuple
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.printing.str import sstr
from sympy.physics.units import (
from sympy.physics.units.util import convert_to, check_dimensions
from sympy.testing.pytest import raises
def test_check_dimensions():
    x = symbols('x')
    assert check_dimensions(inch + x) == inch + x
    assert check_dimensions(length + x) == length + x
    assert check_dimensions((length + x).subs(x, length)) == length
    assert check_dimensions(newton * meter + joule) == joule + meter * newton
    raises(ValueError, lambda: check_dimensions(inch + 1))
    raises(ValueError, lambda: check_dimensions(length + 1))
    raises(ValueError, lambda: check_dimensions(length + time))
    raises(ValueError, lambda: check_dimensions(meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + second))
    raises(ValueError, lambda: check_dimensions(2 * meter + 3 * second))
    raises(ValueError, lambda: check_dimensions(1 / second + 1 / meter))
    raises(ValueError, lambda: check_dimensions(2 * meter * (mile + centimeter) + km))