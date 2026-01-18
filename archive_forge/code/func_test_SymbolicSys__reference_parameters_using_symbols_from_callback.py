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
@requires('sym', 'pygslodeiv2')
@pytest.mark.parametrize('method', ['rkck', 'rk4imp'])
def test_SymbolicSys__reference_parameters_using_symbols_from_callback(method):
    be = sym.Backend('sympy')
    k = be.Symbol('p')

    def dydt(t, y):
        return [-k * y[0]]
    odesys1 = SymbolicSys.from_callback(dydt, 1, backend=be, params=True)
    odesys2 = SymbolicSys.from_callback(dydt, 1, backend=be, par_by_name=True, param_names=[], params=True)
    tout = [0, 1e-09, 1e-07, 1e-05, 0.001, 0.1]
    for symsys in (odesys1, odesys2):
        for y_symb in [False, True]:
            for p_symb in [False, True]:
                xout, yout, info = symsys.integrate(tout, {symsys.dep[0]: 2} if y_symb else [2], {k: 3} if p_symb else [3], method=method, integrator='gsl', atol=1e-12, rtol=1e-12)
                assert xout.size > 4
                assert np.allclose(yout[:, 0], 2 * np.exp(-3 * xout))