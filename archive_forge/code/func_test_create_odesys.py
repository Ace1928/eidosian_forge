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
@requires('numpy', 'pyodesys', 'sympy', 'pycvodes')
def test_create_odesys():
    rsys = ReactionSystem.from_string("\n    A -> B; 'k1'\n    B + C -> P; 'k2'\n    ", substance_factory=Substance)
    odesys, odesys_extra = create_odesys(rsys, unit_registry=SI_base_registry)
    tend_ul = 10
    init_conc_ul = {'A': 1e-06, 'B': 0, 'C': 1}
    params_ul = dict(k1=3, k2=4)
    tend = tend_ul * u.s
    params = {'k1': params_ul['k1'] / u.s, 'k2': params_ul['k2'] / u.M / u.s}
    init_conc = {k: v * u.molar for k, v in init_conc_ul.items()}
    validation = odesys_extra['validate'](dict(init_conc, **params))
    P, = validation['not_seen']
    assert P == 'P'
    ref_rates = {'A': -params['k1'] * init_conc['A'], 'P': params['k2'] * init_conc['B'] * init_conc['C']}
    ref_rates['B'] = -ref_rates['A'] - ref_rates['P']
    ref_rates['C'] = -ref_rates['P']
    assert validation['rates'] == ref_rates
    result1, result1_extra = odesys_extra['unit_aware_solve'](tend, defaultdict(lambda: 0 * u.molar, init_conc), params, integrator='cvode')
    assert result1.info['success']
    result2 = odesys.integrate(tend_ul, defaultdict(lambda: 0, init_conc_ul), params_ul, integrator='cvode')
    assert np.allclose(result2.yout[-1, :], to_unitless(result1.yout[-1, :], u.molar))