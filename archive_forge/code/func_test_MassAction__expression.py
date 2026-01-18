import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
@requires(units_library)
def test_MassAction__expression():

    class GibbsExpr(Expr):
        parameter_keys = ('temperature',)
        argument_names = tuple('dS_over_R dCp_over_R dH_over_R Tref'.split())

        def __call__(self, variables, backend=patched_numpy, **kwargs):
            am = dict(zip(self.argument_names, map(simplified, self.all_args(variables, backend=backend))))
            T, = self.all_params(variables, backend=backend)
            return backend.exp(am['dS_over_R']) * (T / am['Tref']) ** am['dCp_over_R'] * backend.exp(-am['dH_over_R'] / T)
    GeNH3 = GibbsExpr(dict(dS_over_R=18.8 * u.cal / u.K / u.mol / default_constants.molar_gas_constant, dCp_over_R=52 * u.cal / u.K / u.mol / default_constants.molar_gas_constant, dH_over_R=-870.0 * u.cal / u.mol / default_constants.molar_gas_constant, Tref=298.15 * u.K))
    reac_prod = ({'NH3': 1, 'H2O': 1}, {'NH4+': 1, 'OH-': 1})
    Equilibrium(*reac_prod, GeNH3).check_consistent_units(throw=True)
    Equilibrium(*reac_prod[::-1], 1 / GeNH3).check_consistent_units(throw=True)
    Ea = 40000.0 * u.J / u.mol
    R = default_constants.molar_gas_constant
    A, Ea_over_R = (120000000000.0 / u.molar ** 2 / u.second, Ea / R)
    arrh = Arrhenius([A, Ea_over_R])
    ama = MassAction(arrh)
    ma_mul_expr = ama * GeNH3
    ma_div_expr = ama / GeNH3
    expr_mul_ma = GeNH3 * ama
    expr_div_ma = GeNH3 / ama
    assert all((isinstance(expr, MassAction) for expr in [ma_mul_expr, ma_div_expr, expr_mul_ma, expr_div_ma]))
    Reaction(*reac_prod, ama).check_consistent_units(throw=True)
    Reaction(*reac_prod[::-1], 42 * ma_div_expr).check_consistent_units(throw=True)
    Reaction(*reac_prod[::-1], 42 / u.M / u.s / GeNH3).check_consistent_units(throw=True)
    varbls = {'temperature': 298.15 * u.K}
    r_ama, r_GeNH3 = (ama.rate_coeff(varbls), GeNH3(varbls))
    assert allclose(ma_mul_expr.rate_coeff(varbls), r_ama * r_GeNH3)
    assert allclose(ma_div_expr.rate_coeff(varbls), r_ama / r_GeNH3)
    assert allclose(expr_mul_ma.rate_coeff(varbls), r_GeNH3 * r_ama)
    assert allclose(expr_div_ma.rate_coeff(varbls), r_GeNH3 / r_ama)