import warnings
from sympy.core.add import Add
from sympy.core.function import (Function, diff)
from sympy.core.numbers import (Number, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
from sympy.physics.units import (amount_of_substance, area, convert_to, find_unit,
from sympy.physics.units.definitions import (amu, au, centimeter, coulomb,
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units.prefixes import PREFIXES, kilo
from sympy.physics.units.quantities import PhysicalConstant, Quantity
from sympy.physics.units.systems import SI
from sympy.testing.pytest import raises
def test_issue_24062():
    from sympy.core.numbers import E
    from sympy.physics.units import impedance, capacitance, time, ohm, farad, second
    R = Quantity('R')
    C = Quantity('C')
    T = Quantity('T')
    SI.set_quantity_dimension(R, impedance)
    SI.set_quantity_dimension(C, capacitance)
    SI.set_quantity_dimension(T, time)
    R.set_global_relative_scale_factor(1, ohm)
    C.set_global_relative_scale_factor(1, farad)
    T.set_global_relative_scale_factor(1, second)
    expr = T / (R * C)
    dim = SI._collect_factor_and_dimension(expr)[1]
    assert SI.get_dimension_system().is_dimensionless(dim)
    exp_expr = 1 + exp(expr)
    assert SI._collect_factor_and_dimension(exp_expr) == (1 + E, Dimension(1))