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
def test_Species():
    s = Species.from_formula('H2O')
    assert s.phase_idx == 0
    mapping = {'(aq)': 0, '(s)': 1, '(g)': 2}
    assert Species.from_formula('CO2(g)').phase_idx == 3
    assert Species.from_formula('CO2(g)', mapping).phase_idx == 2
    assert Species.from_formula('CO2(aq)', mapping).phase_idx == 0
    assert Species.from_formula('NaCl(s)').phase_idx == 1
    assert Species.from_formula('NaCl(s)', phase_idx=7).phase_idx == 7
    assert Species.from_formula('CO2(aq)', mapping, phase_idx=7).phase_idx == 7
    uranyl_ads = Species.from_formula('UO2+2(ads)', phases={'(aq)': 0, '(ads)': 1})
    assert uranyl_ads.composition == {0: 2, 92: 1, 8: 2}
    assert uranyl_ads.phase_idx == 1