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
def test_balance_stoichiometry__simple():
    reac, prod = balance_stoichiometry({'NH4ClO4', 'Al'}, {'Al2O3', 'HCl', 'H2O', 'N2'})
    assert reac == {'NH4ClO4': 6, 'Al': 10}
    assert prod == {'Al2O3': 5, 'HCl': 6, 'H2O': 9, 'N2': 3}
    r2, p2 = balance_stoichiometry({'Na2CO3'}, {'Na2O', 'CO2'})
    assert r2 == {'Na2CO3': 1}
    assert p2 == {'Na2O': 1, 'CO2': 1}