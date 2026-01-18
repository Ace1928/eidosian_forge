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
@requires('pyodesys', 'scipy', 'sym')
def test_create_odesys__cstr():
    rsys = ReactionSystem.from_string("2 H2O2 -> O2 + 2 H2O; 'k2'")
    fr, fc = ('feedratio', OrderedDict([(sk, 'fc_%s' % sk) for sk in rsys.substances]))
    odesys, extra = create_odesys(rsys, rates_kw=dict(cstr_fr_fc=(fr, fc)))
    _check_cstr(odesys, fr, fc, extra_pars=dict(k2=5))