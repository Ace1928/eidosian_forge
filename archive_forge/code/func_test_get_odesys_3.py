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
def test_get_odesys_3():
    M = u.molar
    s = u.second
    mol = u.mol
    m = u.metre
    substances = list(map(Substance, 'H2O H+ OH-'.split()))
    dissociation = Reaction({'H2O': 1}, {'H+': 1, 'OH-': 1}, 2.47e-05 / s)
    recombination = Reaction({'H+': 1, 'OH-': 1}, {'H2O': 1}, 137000000000.0 / M / s)
    rsys = ReactionSystem([dissociation, recombination], substances)
    odesys = get_odesys(rsys, include_params=True, unit_registry=SI_base_registry, output_conc_unit=M)[0]
    c0 = {'H2O': 55.4 * M, 'H+': 1e-07 * M, 'OH-': 0.0001 * mol / m ** 3}
    x, y, p = odesys.to_arrays(-42 * u.second, rsys.as_per_substance_array(c0, unit=M), ())
    fout = odesys.f_cb(x, y, p)
    time_unit = get_derived_unit(SI_base_registry, 'time')
    conc_unit = get_derived_unit(SI_base_registry, 'concentration')
    r1 = to_unitless(55.4 * 2.47e-05 * M / s, conc_unit / time_unit)
    r2 = to_unitless(1e-14 * 137000000000.0 * M / s, conc_unit / time_unit)
    assert np.all(abs(fout[:, 0] - r2 + r1)) < 1e-10
    assert np.all(abs(fout[:, 1] - r1 + r2)) < 1e-10
    assert np.all(abs(fout[:, 2] - r1 + r2)) < 1e-10