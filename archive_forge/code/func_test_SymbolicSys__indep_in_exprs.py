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
@requires('sym')
def test_SymbolicSys__indep_in_exprs():

    def dydt(t, y, p):
        return [t * p[0] * y[0]]
    be = sym.Backend('sympy')
    t, y, p = map(be.Symbol, 't y p'.split())
    odesys = SymbolicSys([(y, dydt(t, [y], [p])[0])], t, params=True)
    fout = odesys.f_cb(2, [3], [4])
    assert len(fout) == 1
    assert abs(fout[0] - 2 * 3 * 4) < 1e-14