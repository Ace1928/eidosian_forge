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
def refA(t, A0, A, Ea_over_R, T0, dTdt):
    T = T0 + dTdt * t
    d_Ei = sp.Ei(-Ea_over_R / T0).n(100).round(90) - sp.Ei(-Ea_over_R / T).n(100).round(90)
    d_Texp = T0 * sp.exp(-Ea_over_R / T0) - T * sp.exp(-Ea_over_R / T)
    return A0 * sp.exp(A / dTdt * (Ea_over_R * d_Ei + d_Texp)).n(30)