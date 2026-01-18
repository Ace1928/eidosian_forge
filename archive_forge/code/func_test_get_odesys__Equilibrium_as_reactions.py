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
def test_get_odesys__Equilibrium_as_reactions():
    from chempy import Equilibrium, ReactionSystem
    eq = Equilibrium({'Fe+3', 'SCN-'}, {'FeSCN+2'}, 10 ** 2)
    substances = 'Fe+3 SCN- FeSCN+2'.split()
    rsys = ReactionSystem(eq.as_reactions(kf=3.0), substances)
    odesys, extra = get_odesys(rsys)
    init_conc = {'Fe+3': 1.0, 'SCN-': 0.3, 'FeSCN+2': 0}
    tout, Cout, info = odesys.integrate(5, init_conc, integrator='cvode', atol=1e-11, rtol=1e-12)
    cmplx_ref = binary_rev(tout, 3, 3.0 / 100, init_conc['FeSCN+2'], init_conc['Fe+3'], init_conc['SCN-'])
    assert np.allclose(Cout[:, 2], cmplx_ref)