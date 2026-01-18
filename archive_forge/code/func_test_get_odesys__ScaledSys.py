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
def test_get_odesys__ScaledSys():
    from pyodesys.symbolic import ScaledSys
    k = 0.2
    a = Substance('A')
    b = Substance('B')
    r = Reaction({'A': 1}, {'B': 1}, param=k)
    rsys = ReactionSystem([r], [a, b])
    assert sorted(rsys.substances.keys()) == ['A', 'B']
    odesys = get_odesys(rsys, include_params=True, SymbolicSys=ScaledSys)[0]
    c0 = {'A': 1.0, 'B': 3.0}
    t = np.linspace(0.0, 10.0)
    xout, yout, info = odesys.integrate(t, c0)
    yref = np.zeros((t.size, 2))
    yref[:, 0] = np.exp(-k * t)
    yref[:, 1] = 4 - np.exp(-k * t)
    assert np.allclose(yout, yref)