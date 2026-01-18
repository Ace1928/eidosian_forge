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
@requires('pycvodes', 'sym', units_library)
def test_create_odesys__validate__catalyst():
    rsys1 = ReactionSystem.from_string("\n    H2O2 + Pt -> 2 OH + Pt; 'k_decomp'\n    ")
    ic1 = defaultdict(lambda: 0 * u.molar, {'H2O2': 3.0 * u.molar, 'Pt': 0.5 * u.molar})
    t1 = linspace(0 * u.s, 0.3 * u.s, 7)
    p1 = dict(k_decomp=42 / u.second / u.molar)
    odesys1, odesys_extra = create_odesys(rsys1)
    validation = odesys_extra['validate'](dict(ic1, **p1))
    assert validation['not_seen'] == {'OH'}
    dedim_ctx = _mk_dedim(SI_base_registry)
    (t, c, _p), dedim_extra = dedim_ctx['dedim_tcp'](t1, [ic1[k] for k in odesys1.names], p1)
    result1 = odesys1.integrate(t, c, _p)
    tout = result1.xout * dedim_extra['unit_time']
    cout = result1.yout * dedim_extra['unit_conc']
    yref1 = ic1['H2O2'] * np.exp(-tout * ic1['Pt'] * p1['k_decomp'])
    assert allclose(yref1, cout[:, odesys1.names.index('H2O2')], rtol=1e-06)