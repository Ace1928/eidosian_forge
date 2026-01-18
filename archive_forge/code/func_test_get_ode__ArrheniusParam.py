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
@requires('pyodesys')
def test_get_ode__ArrheniusParam():
    rxn = Reaction({'A': 1}, {'B': 1}, None)
    rxn.param = ArrheniusParam(10000000000.0, 40000.0)
    rsys = ReactionSystem([rxn], 'A B')
    odesys = get_odesys(rsys, include_params=True)[0]
    conc = {'A': 3, 'B': 5}
    x, y, p = odesys.to_arrays(-37, conc, {'temperature': 200})
    fout = odesys.f_cb(x, y, p)
    ref = 3 * 10000000000.0 * np.exp(-40000.0 / 8.314472 / 200)
    assert np.all(abs((fout[:, 0] + ref) / ref) < 1e-14)
    assert np.all(abs((fout[:, 1] - ref) / ref) < 1e-14)