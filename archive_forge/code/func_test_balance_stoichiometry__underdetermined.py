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
def test_balance_stoichiometry__underdetermined():
    try:
        from pulp import PulpSolverError
    except ModuleNotFoundError:
        from pulp.solvers import PulpSolverError
    with pytest.raises(ValueError):
        balance_stoichiometry({'C2H6', 'O2'}, {'H2O', 'CO2', 'CO'}, underdetermined=False)
    reac, prod = balance_stoichiometry({'C2H6', 'O2'}, {'H2O', 'CO2', 'CO'})
    r1 = {'C7H5O3-', 'O2', 'C21H27N7O14P2-2', 'H+'}
    p1 = {'C7H5O4-', 'C21H26N7O14P2-', 'H2O'}
    bal1 = balance_stoichiometry(r1, p1, underdetermined=None)
    assert bal1 == ({'C21H27N7O14P2-2': 1, 'H+': 1, 'C7H5O3-': 1, 'O2': 1}, {'C21H26N7O14P2-': 1, 'H2O': 1, 'C7H5O4-': 1})
    with pytest.raises(ValueError):
        balance_stoichiometry({'C3H4O3', 'H3PO4'}, {'C3H6O3'}, underdetermined=None)
    for underdet in [False, True, None]:
        with pytest.raises((ValueError, PulpSolverError)):
            balance_stoichiometry({'C3H6O3'}, {'C3H4O3'}, underdetermined=underdet)
    with pytest.raises(ValueError):
        balance_stoichiometry({'C21H36N7O16P3S', 'C3H4O3'}, {'H2O', 'C5H8O3', 'C24H38N7O18P3S'})