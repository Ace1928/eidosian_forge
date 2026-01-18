from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
@requires('sympy')
def test_Equilibrium__eliminate():
    e1 = Equilibrium({'A': 1, 'B': 2}, {'C': 3})
    e2 = Equilibrium({'D': 5, 'B': 7}, {'E': 11})
    coeff = Equilibrium.eliminate([e1, e2], 'B')
    assert coeff == [7, -2]
    e3 = coeff[0] * e1 + coeff[1] * e2
    assert e3.net_stoich('B') == (0,)
    e4 = e1 * coeff[0] + coeff[1] * e2
    assert e4.net_stoich('B') == (0,)
    assert (-e1).reac == {'C': 3}
    assert (e2 * -3).reac == {'E': 33}