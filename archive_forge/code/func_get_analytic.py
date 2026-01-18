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
def get_analytic(result, k, n):
    ref = binary_irrev_cstr(result.xout, 5, result.named_dep('H2O2')[0], result.named_dep(k)[0], result.named_param(fc['H2O2']), result.named_param(fc[k]), result.named_param(fr), n)
    return np.array(ref).T