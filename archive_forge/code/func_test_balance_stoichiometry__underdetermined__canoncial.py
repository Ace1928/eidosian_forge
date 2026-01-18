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
@requires('sympy', 'pulp')
def test_balance_stoichiometry__underdetermined__canoncial():
    r2 = {'O2', 'O3', 'C', 'NO', 'N2O', 'NO2', 'N2O4'}
    p2 = {'CO', 'CO2', 'N2'}
    bal2 = balance_stoichiometry(r2, p2, underdetermined=None)
    ref2 = ({'O2': 1, 'O3': 1, 'C': 7, 'NO': 1, 'N2O': 1, 'NO2': 1, 'N2O4': 1}, {'CO': 1, 'CO2': 6, 'N2': 3})
    substances = {k: Substance.from_formula(k) for k in r2 | p2}
    assert all((viol == 0 for viol in Reaction(*ref2).composition_violation(substances)))
    assert sum(bal2[0].values()) + sum(bal2[1].values()) <= sum(ref2[0].values()) + sum(ref2[1].values())
    assert bal2 == ref2