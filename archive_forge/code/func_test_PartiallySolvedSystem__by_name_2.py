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
def test_PartiallySolvedSystem__by_name_2():
    yn, pn = ('x y z'.split(), 'p q r'.split())
    odesys = _get_decay3_names(yn, pn)
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0, be: {odesys['x']: y0[odesys['x']] * be.exp(-p0['p'] * (odesys.indep - x0))})
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 1.5]

    def _check(res):
        ref = np.array(bateman_full(y0, k, res.xout - res.xout[0], exp=np.exp)).T
        assert res.info['success']
        assert np.allclose(res.yout, ref)
    args = ([0, 1], dict(zip(yn, y0)), dict(zip(pn, k)))
    kwargs = dict(integrator='cvode')
    _check(odesys.integrate(*args, **kwargs))
    _check(partsys.integrate(*args, **kwargs))
    scaledsys = ScaledSys.from_other(partsys, dep_scaling=42, indep_scaling=17)
    _check(scaledsys.integrate(*args, **kwargs))