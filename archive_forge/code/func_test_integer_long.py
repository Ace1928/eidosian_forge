from symengine.test_utilities import raises
from symengine import Integer, I, S, Symbol, pi, Rational
from symengine.lib.symengine_wrapper import (perfect_power, is_square, integer_nthroot)
def test_integer_long():
    py_int = 123434444444444444444
    i = Integer(py_int)
    assert str(i) == str(py_int)
    assert int(i) == py_int