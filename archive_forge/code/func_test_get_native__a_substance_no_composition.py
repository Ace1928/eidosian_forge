from collections import defaultdict
from functools import reduce
from operator import mul
import pytest
from chempy import ReactionSystem
from chempy.units import (
from chempy.util.testing import requires
from ..integrated import binary_rev
from ..ode import get_odesys
from .._native import get_native
@requires('pygslodeiv2', 'pyodesys')
@pytest.mark.parametrize('solve', [(), ('H2O',)])
def test_get_native__a_substance_no_composition(solve):
    rsys = ReactionSystem.from_string('\n'.join(['H2O -> H2O+ + e-(aq); 1e-8', 'e-(aq) + H2O+ -> H2O; 1e10']))
    odesys, extra = get_odesys(rsys)
    c0 = {'H2O': 0, 'H2O+': 2e-09, 'e-(aq)': 3e-09}
    if len(solve) > 0:
        from pyodesys.symbolic import PartiallySolvedSystem
        odesys = PartiallySolvedSystem(odesys, extra['linear_dependencies'](solve))
    odesys = get_native(rsys, odesys, 'gsl')
    xout, yout, info = odesys.integrate(1, c0, atol=1e-15, rtol=1e-15, integrator='gsl')
    c_reac = (c0['H2O+'], c0['e-(aq)'])
    H2O_ref = binary_rev(xout, 10000000000.0, 0.0001, c0['H2O'], max(c_reac), min(c_reac))
    assert np.allclose(yout[:, odesys.names.index('H2O')], H2O_ref)
    assert np.allclose(yout[:, odesys.names.index('H2O+')], c0['H2O+'] + c0['H2O'] - H2O_ref)
    assert np.allclose(yout[:, odesys.names.index('e-(aq)')], c0['e-(aq)'] + c0['H2O'] - H2O_ref)