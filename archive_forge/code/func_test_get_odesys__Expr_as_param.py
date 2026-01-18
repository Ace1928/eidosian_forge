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
def test_get_odesys__Expr_as_param():

    def _eyring_pe(args, T, backend=math, **kwargs):
        freq, = args
        return freq * T
    EyringPreExp = Expr.from_callback(_eyring_pe, argument_names=('freq',), parameter_keys=('temperature',))

    def _k(args, T, backend=math, **kwargs):
        A, H, S = args
        return A * backend.exp(-(H - T * S) / (8.314511 * T))
    EyringMA = MassAction.from_callback(_k, parameter_keys=('temperature',), argument_names=('Aa', 'Ha', 'Sa'))
    kb_h = 20800000000.0
    rxn = Reaction({'A'}, {'B'}, EyringMA(unique_keys=('A_u', 'H_u', 'S_u')))
    rsys = ReactionSystem([rxn], ['A', 'B'])
    odesys, extra = get_odesys(rsys, include_params=False, substitutions={'A_u': EyringPreExp(kb_h)})
    y0 = defaultdict(float, {'A': 7.0})
    rt = 293.15
    xout, yout, info = odesys.integrate(5, y0, {'H_u': 117000.0, 'S_u': 150, 'temperature': rt}, integrator='cvode', atol=1e-12, rtol=1e-10, nsteps=1000)
    kref = kb_h * rt * np.exp(-(117000.0 - rt * 150) / (8.314511 * rt))
    ref = y0['A'] * np.exp(-kref * xout)
    assert np.allclose(yout[:, 0], ref)
    assert np.allclose(yout[:, 1], y0['A'] - ref)