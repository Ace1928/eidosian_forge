import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
@requires(units_library)
def test_GibbsEqConst__nested():

    class TExpr(Expr):
        argument_names = ('heat_capacity',)
        parameter_keys = ('energy',)

        def __call__(self, variables, backend=None):
            heat_capacity, = self.all_args(variables, backend=backend)
            energy, = self.all_params(variables, backend=backend)
            return energy / heat_capacity
    R = 8.314 * du.J / du.K / du.mol
    T = TExpr([10.0 * du.J / du.K])
    dH, dS = (-4000.0 * du.J / du.mol, 16 * du.J / du.K / du.mol)
    gee = GibbsEqConst([dH / R, dS / R])
    be = Backend()
    Tref = 298.15 * du.K
    ref = be.exp(-(dH - Tref * dS) / (R * Tref))
    assert be.abs((gee.eq_const({'energy': 2981.5 * du.J, 'temperature': T}, backend=be) - ref) / ref) < 1e-14