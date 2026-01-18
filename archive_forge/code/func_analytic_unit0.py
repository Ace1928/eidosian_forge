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
def analytic_unit0(t, k, m, dH, dS):
    R = 8.314472
    kB = 1.3806504e-23
    h = 6.62606896e-34
    A = kB / h * np.exp(dS / R)
    B = dH / R
    return k * np.exp(B * (k * t + 2 * m) / (m * (k * t + m))) / (A * (-B ** 2 * np.exp(B / (k * t + m)) * expi(-B / (k * t + m)) - B * k * t - B * m + k ** 2 * t ** 2 + 2 * k * m * t + m ** 2) * np.exp(B / m) + (A * B ** 2 * np.exp(B / m) * expi(-B / m) - A * m * (-B + m) + k * np.exp(B / m)) * np.exp(B / (k * t + m)))