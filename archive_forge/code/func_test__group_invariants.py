from __future__ import print_function, absolute_import, division
from collections import defaultdict, OrderedDict
from itertools import product
import math
import numpy as np
import pytest
from .. import ODESys
from ..core import integrate_auto_switch, chained_parameter_variation
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires, pycvodes_double, pycvodes_klu
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa
@requires('sym', 'sympy')
def test__group_invariants():
    be = sym.Backend('sympy')
    x, y, z = symbs = be.symbols('x y z')
    coeff1 = [3, 2, -1]
    expr1 = 3 * x + 2 * y - z
    lin, nonlin = _group_invariants([expr1], symbs, be)
    assert lin == [coeff1]
    assert nonlin == []
    expr2 = 3 * x * x + 2 * y - z
    lin, nonlin = _group_invariants([expr2], symbs, be)
    assert lin == []
    assert nonlin == [expr2]
    lin, nonlin = _group_invariants([expr1, expr2], symbs, be)
    assert lin == [coeff1]
    assert nonlin == [expr2]
    lin, nonlin = _group_invariants([x + be.exp(y)], symbs, be)
    assert lin == []
    assert nonlin == [x + be.exp(y)]