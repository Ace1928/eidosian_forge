from sympy.core.containers import Tuple
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.printing.str import sstr
from sympy.physics.units import (
from sympy.physics.units.util import convert_to, check_dimensions
from sympy.testing.pytest import raises
def test_convert_to_quantities():
    assert convert_to(3, meter) == 3
    assert convert_to(mile, kilometer) == 25146 * kilometer / 15625
    assert convert_to(meter / second, speed_of_light) == speed_of_light / 299792458
    assert convert_to(299792458 * meter / second, speed_of_light) == speed_of_light
    assert convert_to(2 * 299792458 * meter / second, speed_of_light) == 2 * speed_of_light
    assert convert_to(speed_of_light, meter / second) == 299792458 * meter / second
    assert convert_to(2 * speed_of_light, meter / second) == 599584916 * meter / second
    assert convert_to(day, second) == 86400 * second
    assert convert_to(2 * hour, minute) == 120 * minute
    assert convert_to(mile, meter) == 201168 * meter / 125
    assert convert_to(mile / hour, kilometer / hour) == 25146 * kilometer / (15625 * hour)
    assert convert_to(3 * newton, meter / second) == 3 * newton
    assert convert_to(3 * newton, kilogram * meter / second ** 2) == 3 * meter * kilogram / second ** 2
    assert convert_to(kilometer + mile, meter) == 326168 * meter / 125
    assert convert_to(2 * kilometer + 3 * mile, meter) == 853504 * meter / 125
    assert convert_to(inch ** 2, meter ** 2) == 16129 * meter ** 2 / 25000000
    assert convert_to(3 * inch ** 2, meter) == 48387 * meter ** 2 / 25000000
    assert convert_to(2 * kilometer / hour + 3 * mile / hour, meter / second) == 53344 * meter / (28125 * second)
    assert convert_to(2 * kilometer / hour + 3 * mile / hour, centimeter / second) == 213376 * centimeter / (1125 * second)
    assert convert_to(kilometer * (mile + kilometer), meter) == 2609344 * meter ** 2
    assert convert_to(steradian, coulomb) == steradian
    assert convert_to(radians, degree) == 180 * degree / pi
    assert convert_to(radians, [meter, degree]) == 180 * degree / pi
    assert convert_to(pi * radians, degree) == 180 * degree
    assert convert_to(pi, degree) == 180 * degree