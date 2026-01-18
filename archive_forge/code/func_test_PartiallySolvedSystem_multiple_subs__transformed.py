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
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem_multiple_subs__transformed(integrator):
    odesys = _get_decay3(lower_bounds=[0, 0, 0])

    def substitutions(x0, y0, p0, be):
        analytic0 = y0[0] * be.exp(-p0[0] * (odesys.indep - x0))
        analytic2 = y0[0] + y0[1] + y0[2] - analytic0 - odesys.dep[1]
        return {odesys.dep[0]: analytic0, odesys.dep[2]: analytic2}
    partsys = PartiallySolvedSystem(odesys, substitutions)
    LogLogSys = symmetricsys(get_logexp(), get_logexp())
    loglogpartsys = LogLogSys.from_other(partsys)
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 0]
    tend = 1
    for system, ny_internal in [(odesys, 3), (partsys, 1), (loglogpartsys, 1)]:
        xout, yout, info = system.integrate([1e-12, tend], y0, k, integrator=integrator, first_step=1e-14, nsteps=1000)
        ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
        assert info['success']
        assert info['internal_yout'].shape[-1] == ny_internal
        if system == loglogpartsys:
            assert info['internal_yout'][-1, 0] < 0
        assert np.allclose(yout, ref)