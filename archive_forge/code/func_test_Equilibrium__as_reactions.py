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
@requires(units_library)
def test_Equilibrium__as_reactions():
    s = default_units.second
    M = default_units.molar
    H2O, Hp, OHm = map(Substance, 'H2O H+ OH-'.split())
    eq = Equilibrium({'H2O': 1}, {'H+': 1, 'OH-': 1}, 1e-14)
    rate = 131000000000.0 / M / s
    fw, bw = eq.as_reactions(kb=rate, units=default_units)
    assert abs((bw.param - rate) / rate) < 1e-15
    assert abs(fw.param / M / bw.param - 1e-14) / 1e-14 < 1e-15