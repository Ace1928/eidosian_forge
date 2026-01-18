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
def test_SpecialFraction():
    k, kprime = (3.142, 2.718)
    rsys = _get_SpecialFraction_rsys(k, kprime)
    odesys = get_odesys(rsys, include_params=True)[0]
    c0 = {'H2': 13, 'Br2': 17, 'HBr': 19}
    r = k * c0['H2'] * c0['Br2'] ** (3 / 2) / (c0['Br2'] + kprime * c0['HBr'])
    ref = rsys.as_per_substance_array({'H2': -r, 'Br2': -r, 'HBr': 2 * r})
    res = odesys.f_cb(0, rsys.as_per_substance_array(c0))
    assert np.allclose(res, ref)