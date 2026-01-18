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
@requires('numpy', 'pyodesys')
def test_get_odesys_2():
    g = Radiolytic([3.14])
    a = Substance('A')
    b = Substance('B')
    r = Reaction({'A': 1}, {'B': 1}, param=g)
    rsys = ReactionSystem([r], [a, b])
    odesys = get_odesys(rsys, include_params=True)[0]
    c0 = {'A': 1.0, 'B': 3.0}
    t = np.linspace(0.0, 0.1)
    xout, yout, info = odesys.integrate(t, rsys.as_per_substance_array(c0), {'doserate': 2.72, 'density': 0.998})
    yref = np.zeros((t.size, 2))
    k = 3.14 * 2.72 * 0.998
    yref[:, 0] = 1 - k * t
    yref[:, 1] = 3 + k * t
    assert np.allclose(yout, yref)