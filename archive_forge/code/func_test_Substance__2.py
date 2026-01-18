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
def test_Substance__2():
    H2O = Substance(name='H2O', charge=0, latex_name='\\mathrm{H_{2}O}', data={'pKa': 14})
    OH_m = Substance(name='OH-', charge=-1, latex_name='\\mathrm{OH^{-}}')
    assert sorted([OH_m, H2O], key=attrgetter('name')) == [H2O, OH_m]