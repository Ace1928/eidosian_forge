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
def test_balance_stoichiometry__substances__underdetermined():
    substances = {s.name: s for s in [Substance('eggs_6pack', composition=dict(eggs=6)), Substance('milk_carton', composition=dict(cups_of_milk=4)), Substance('flour_bag', composition=dict(spoons_of_flour=30)), Substance('pancake', composition=dict(eggs=1, cups_of_milk=1, spoons_of_flour=2)), Substance('waffle', composition=dict(eggs=2, cups_of_milk=2, spoons_of_flour=3))]}
    ur1 = {'eggs_6pack', 'milk_carton', 'flour_bag'}
    up1 = {'pancake', 'waffle'}
    br1, bp1 = balance_stoichiometry(ur1, up1, substances=substances, underdetermined=None)
    ref_r1 = {'eggs_6pack': 6, 'flour_bag': 2, 'milk_carton': 9}
    ref_p1 = {'pancake': 12, 'waffle': 12}
    assert all((viol == 0 for viol in Reaction(ref_r1, ref_p1).composition_violation(substances)))
    assert all((v > 0 for v in br1.values())) and all((v > 0 for v in bp1.values()))
    assert bp1 == ref_p1
    assert br1 == ref_r1