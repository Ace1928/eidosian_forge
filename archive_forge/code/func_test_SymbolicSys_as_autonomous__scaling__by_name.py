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
@requires('sym', 'pycvodes')
def test_SymbolicSys_as_autonomous__scaling__by_name():

    def get_odesys(scaling=1):

        def rhs(t, y, p, backend=math):
            R = 8.314
            T = 300 + 10 * backend.sin(0.2 * math.pi * t - math.pi / 2)
            kB_h = 20836600000.0
            k1 = kB_h * T * backend.exp(p['dS1'] / R - p['dH1'] / (R * T)) / scaling
            k2 = kB_h * T * backend.exp(p['dS2'] / R - p['dH2'] / (R * T)) / scaling
            r1 = k1 * y['HNO2'] ** 2
            r2 = k2 * y['NO2'] ** 2
            return {'HNO2': -2 * r1, 'H2O': r1, 'NO': r1, 'NO2': r1 - 2 * r2, 'N2O4': r2}
        return SymbolicSys.from_callback(rhs, 5, 4, names='HNO2 H2O NO NO2 N2O4'.split(), param_names='dH1 dS1 dH2 dS2'.split(), dep_by_name=True, par_by_name=True, to_arrays_callbacks=(None, lambda y: [_y * scaling for _y in y], None))

    def check(system):
        init_y = {'HNO2': 1, 'H2O': 55, 'NO': 0, 'NO2': 0, 'N2O4': 0}
        p = {'dH1': 85000.0, 'dS1': 10, 'dH2': 70000.0, 'dS2': 20}
        return system.integrate(np.linspace(0, 60, 200), init_y, p, integrator='cvode', nsteps=5000)

    def compare_autonomous(scaling):
        odesys = get_odesys(scaling)
        autsys = odesys.as_autonomous()
        copsys = SymbolicSys.from_other(autsys)
        res1 = check(odesys)
        res2 = check(autsys)
        res3 = check(copsys)
        assert np.allclose(res1.yout, res2.yout, atol=1e-06)
        assert np.allclose(res1.yout, res3.yout, atol=1e-06)
    compare_autonomous(1)
    compare_autonomous(1000)