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
@requires(parsing_library)
def test_Substance__from_formula():
    H2O = Substance.from_formula('H2O')
    assert H2O.composition == {1: 2, 8: 1}
    assert H2O.latex_name == 'H_{2}O'
    assert H2O.unicode_name == u'H₂O'
    assert H2O.html_name == u'H<sub>2</sub>O'