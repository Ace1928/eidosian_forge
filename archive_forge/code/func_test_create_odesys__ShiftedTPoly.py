from collections import defaultdict, OrderedDict
from itertools import permutations
import math
import pytest
from chempy import Equilibrium, Reaction, ReactionSystem, Substance
from chempy.thermodynamics.expressions import MassActionEq
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.testing import requires
from .test_rates import _get_SpecialFraction_rsys
from ..arrhenius import ArrheniusParam
from ..rates import Arrhenius, MassAction, Radiolytic, RampedTemp
from .._rates import ShiftedTPoly
from ..ode import (
from ..integrated import dimerization_irrev, binary_rev
@requires('pyodesys', units_library)
def test_create_odesys__ShiftedTPoly():
    rxn = Reaction({'A': 1, 'B': 1}, {'C': 3, 'D': 2}, 'k_bi', {'A': 3})
    rsys = ReactionSystem([rxn], 'A B C D')
    _k0, _k1, T0C = (10, 2, 273.15)
    rate = MassAction(ShiftedTPoly([T0C * u.K, _k0 / u.molar / u.s, _k1 / u.molar / u.s / u.K]))
    T_C = 25
    T = (T0C + T_C) * u.kelvin
    p1 = rate.rate_coeff({'temperature': T})
    assert allclose(p1, (_k0 + _k1 * T_C) / u.molar / u.s)
    odesys, odesys_extra = create_odesys(rsys)
    ics = {'A': 3 * u.molar, 'B': 5 * u.molar, 'C': 11 * u.molar, 'D': 13 * u.molar}
    pars = dict(k_bi=p1)
    validation = odesys_extra['validate'](dict(ics, **pars))
    assert set(map(str, validation['not_seen'])) == {'C', 'D'}
    dedim_ctx = _mk_dedim(SI_base_registry)
    (t, c, _p), dedim_extra = dedim_ctx['dedim_tcp'](-37 * u.s, [ics[k] for k in odesys.names], pars)
    fout = odesys.f_cb(t, c, [_p[pk] for pk in odesys.param_names])
    r = 3 * 5 * (_k0 + _k1 * 25) * 1000
    ref = [-4 * r, -r, 3 * r, 2 * r]
    assert np.all(abs((fout - ref) / ref) < 1e-14)
    odesys.integrate(t, c, _p)