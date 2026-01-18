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
def test_SymbolicSys_from_other_new_params_by_name():
    yn, pn = ('x y z'.split(), 'p q r'.split())
    odesys = _get_decay3_names(yn, pn)
    assert len(odesys.params) == 3
    newode, extra = SymbolicSys.from_other_new_params_by_name(odesys, OrderedDict([('p', lambda x, y, p, backend: p['s'] - 1), ('q', lambda x, y, p, backend: p['s'] + 1)]), ('s',))
    assert len(newode.params) == 2
    tout = np.array([0.3, 0.4, 0.7, 0.9, 1.3, 1.7, 1.8, 2.1])
    y0 = {'x': 7, 'y': 5, 'z': 2}
    p_vals = {'r': 5, 's': 3}
    res1 = newode.integrate(tout, y0, p_vals)
    k = [p_vals['s'] - 1, p_vals['s'] + 1, p_vals['r']]
    ref1 = np.array(bateman_full([y0[n] for n in newode.names], [p_vals['s'] - 1, p_vals['s'] + 1, p_vals['r']], res1.xout - res1.xout[0], exp=np.exp)).T
    assert np.allclose(res1.yout, ref1)
    orip = extra['recalc_params'](res1.xout, res1.yout, res1.params)
    assert np.allclose(orip, np.atleast_2d(k))