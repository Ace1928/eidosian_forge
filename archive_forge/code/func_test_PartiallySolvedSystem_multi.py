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
def test_PartiallySolvedSystem_multi(integrator):
    odesys = _get_decay3()

    def _get_analytic(x0, y0, p0, be):
        a0 = y0[0] * be.exp(-p0[0] * (odesys.indep - x0))
        a1 = y0[0] + y0[1] + y0[2] - a0 - odesys.dep[2]
        return [a0, a1]

    def subst(x0, y0, p0, be):
        a0, a1 = _get_analytic(x0, y0, p0, be)
        return {odesys.dep[0]: a0, odesys.dep[1]: a1}
    partsys = PartiallySolvedSystem(odesys, subst)
    a0, a1 = _get_analytic(partsys.init_indep, partsys.init_dep, odesys.params, odesys.be)
    assert partsys.ny == 1
    assert partsys.exprs[0].subs(odesys.params[2], 0) - odesys.params[1] * a1 == 0