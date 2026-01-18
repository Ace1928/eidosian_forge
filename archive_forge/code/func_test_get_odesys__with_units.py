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
@requires(units_library, 'pyodesys')
def test_get_odesys__with_units():
    a = Substance('A')
    b = Substance('B')
    molar = u.molar
    second = u.second
    r = Reaction({'A': 2}, {'B': 1}, param=0.001 / molar / second)
    rsys = ReactionSystem([r], [a, b])
    odesys = get_odesys(rsys, include_params=True, unit_registry=SI_base_registry)[0]
    c0 = {'A': 13 * u.mol / u.metre ** 3, 'B': 0.2 * u.molar}
    conc_unit = get_derived_unit(SI_base_registry, 'concentration')
    t = np.linspace(0, 10) * u.hour
    xout, yout, info = odesys.integrate(t, rsys.as_per_substance_array(c0, unit=conc_unit), atol=1e-10, rtol=1e-12)
    t_unitless = to_unitless(xout, u.second)
    Aref = dimerization_irrev(t_unitless, 1e-06, 13.0)
    yref = np.zeros((xout.size, 2))
    yref[:, 0] = Aref
    yref[:, 1] = 200 + (13 - Aref) / 2
    assert allclose(yout, yref * conc_unit)