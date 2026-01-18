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
def test_get_ode__TPoly():
    rate = MassAction(ShiftedTPoly([273.15 * u.K, 10 / u.molar / u.s, 2 / u.molar / u.s / u.K]))
    rxn = Reaction({'A': 1, 'B': 1}, {'C': 3, 'D': 2}, rate, {'A': 3})
    rsys = ReactionSystem([rxn], 'A B C D')
    odesys = get_odesys(rsys, unit_registry=SI_base_registry)[0]
    conc = {'A': 3 * u.molar, 'B': 5 * u.molar, 'C': 11 * u.molar, 'D': 13 * u.molar}
    x, y, p = odesys.to_arrays(-37 * u.second, conc, {'temperature': 298.15 * u.kelvin})
    fout = odesys.f_cb(x, y, p)
    r = 3 * 5 * (10 + 2 * 25) * 1000
    ref = [-4 * r, -r, 3 * r, 2 * r]
    assert np.all(abs((fout - ref) / ref) < 1e-14)