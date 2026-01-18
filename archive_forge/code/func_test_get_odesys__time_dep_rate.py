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
def test_get_odesys__time_dep_rate():

    class RampedRate(Expr):
        argument_names = ('rate_constant', 'ramping_rate')

        def __call__(self, variables, reaction, backend=math):
            rate_constant, ramping_rate = self.all_args(variables, backend=backend)
            return rate_constant * ramping_rate * variables['time']
    rate = MassAction(RampedRate([7, 2]))
    rxn = Reaction({'A': 1}, {'B': 3}, rate)
    rsys = ReactionSystem([rxn], 'A B')
    odesys = get_odesys(rsys)[0]
    conc = {'A': 3, 'B': 11}
    x, y, p = odesys.to_arrays([5, 13, 17], conc, ())
    fout = odesys.f_cb(x, y, p)
    r = 2 * 7 * 3
    ref = np.array([[-r * 5, -r * 13, -r * 17], [r * 5 * 3, r * 13 * 3, r * 17 * 3]]).T
    assert np.allclose(fout, ref)