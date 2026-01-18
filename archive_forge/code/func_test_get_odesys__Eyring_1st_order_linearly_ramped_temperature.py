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
@requires('pycvodes', 'sym', 'scipy', units_library)
def test_get_odesys__Eyring_1st_order_linearly_ramped_temperature():
    from scipy.special import expi

    def analytic_unit0(t, T0, dH, dS):
        R = 8.314472
        kB = 1.3806504e-23
        h = 6.62606896e-34
        A = kB / h * np.exp(dS / R)
        B = dH / R
        return np.exp(A * ((-B ** 2 * np.exp(B / T0) * expi(-B / T0) - T0 * (B - T0)) * np.exp(-B / T0) + (B ** 2 * np.exp(B / (t + T0)) * expi(-B / (t + T0)) - (t + T0) * (-B + t + T0)) * np.exp(-B / (t + T0))) / 2)
    T_K = 290
    dH = 80000.0
    dS = 10
    rsys1 = ReactionSystem.from_string('\n    NOBr -> NO + Br; EyringParam(dH={dH}*J/mol, dS={dS}*J/K/mol)\n    '.format(dH=dH, dS=dS))
    NOBr0_M = 0.7
    init_cond = dict(NOBr=NOBr0_M * u.M, NO=0 * u.M, Br=0 * u.M)
    t = 20 * u.second

    def check(rsys):
        odes, extra = get_odesys(rsys, unit_registry=SI_base_registry, constants=const, substitutions={'temperature': RampedTemp([T_K * u.K, 1 * u.K / u.s])})
        for odesys in [odes, odes.as_autonomous()]:
            res = odesys.integrate(t, init_cond, integrator='cvode')
            t_sec = to_unitless(res.xout, u.second)
            NOBr_ref = NOBr0_M * analytic_unit0(t_sec, T_K, dH, dS)
            cmp = to_unitless(res.yout, u.M)
            ref = np.empty_like(cmp)
            ref[:, odesys.names.index('NOBr')] = NOBr_ref
            ref[:, odesys.names.index('Br')] = NOBr0_M - NOBr_ref
            ref[:, odesys.names.index('NO')] = NOBr0_M - NOBr_ref
            assert np.allclose(cmp, ref)
    check(rsys1)
    rsys2 = ReactionSystem.from_string('\n    NOBr -> NO + Br; MassAction(EyringHS([{dH}*J/mol, {dS}*J/K/mol]))\n    '.format(dH=dH, dS=dS))
    check(rsys2)