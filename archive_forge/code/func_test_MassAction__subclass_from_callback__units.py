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
def test_MassAction__subclass_from_callback__units():

    def rate_coeff(variables, all_args, backend, **kwargs):
        return all_args[0] * backend.exp(all_args[1] / variables['temperature'])
    CustomMassAction = MassAction.subclass_from_callback(rate_coeff, cls_attrs=dict(parameter_keys=('temperature',), nargs=2))
    k1 = CustomMassAction([21000000000.0 / u.molar ** 2 / u.second, -5132.2 * u.kelvin])
    rxn = Reaction({'H2': 2, 'O2': 1}, {'H2O': 2}, k1)
    variables = {'temperature': 491.67 * u.rankine, 'H2': 7000 * u.mol / u.metre ** 3, 'O2': 13 * u.molar}
    cma = rxn.rate_expr()
    res = cma(variables, backend=Backend(), reaction=rxn)
    ref = 7 * 7 * 13 * 21000000000.0 * math.exp(-5132.2 / 273.15) * u.molar / u.second
    assert allclose(res, ref)