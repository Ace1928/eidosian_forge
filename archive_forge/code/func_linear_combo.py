from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_to_gdp_transformation import (
from pyomo.core import Constraint, NonNegativeIntegers, Var
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
@transBlock.Constraint(range(dimension))
def linear_combo(b, i):
    return pw_expr.args[i] == sum((pt[i] * transBlock.lambdas[j] for j, pt in enumerate(pw_linear_func._points)))