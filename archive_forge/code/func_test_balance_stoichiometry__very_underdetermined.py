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
def test_balance_stoichiometry__very_underdetermined():
    r3 = set('O2 Fe Al Cr'.split())
    p3 = set('FeO Fe2O3 Fe3O4 Al2O3 Cr2O3 CrO3'.split())
    bal3 = balance_stoichiometry(r3, p3, underdetermined=None)
    ref3 = ({'Fe': 7, 'Al': 2, 'Cr': 3, 'O2': 9}, {k: 2 if k == 'FeO' else 1 for k in p3})
    substances = {k: Substance.from_formula(k) for k in r3 | p3}
    assert all((viol == 0 for viol in Reaction(*ref3).composition_violation(substances)))
    assert sum(bal3[0].values()) + sum(bal3[1].values()) <= sum(ref3[0].values()) + sum(ref3[1].values())
    assert bal3 == ref3