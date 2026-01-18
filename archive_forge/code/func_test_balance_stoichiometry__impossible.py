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
@pytest.mark.parametrize('underdet', [False, None, True])
def test_balance_stoichiometry__impossible(underdet):
    try:
        from pulp import PulpSolverError
    except ModuleNotFoundError:
        from pulp.solvers import PulpSolverError
    with pytest.raises((ValueError, PulpSolverError)):
        r1, p1 = balance_stoichiometry({'CO'}, {'CO2'}, underdetermined=underdet)