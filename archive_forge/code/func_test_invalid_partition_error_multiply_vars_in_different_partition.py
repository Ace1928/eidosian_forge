import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
def test_invalid_partition_error_multiply_vars_in_different_partition(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-10, 10))
    m.y = Var(bounds=(-60, 56))
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x ** 2 + m.x * m.y + m.y ** 2 <= 32)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x ** 2 + m.y ** 2 <= 3)
    m.disjunction = Disjunction(expr=[m.d1, m.d2])
    with self.assertRaisesRegex(GDP_Error, "Variables 'x' and 'y' are multiplied in Constraint 'd1.c', but they are in different partitions! Please ensure that all the constraints in the disjunction are additively separable with respect to the specified partition."):
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x], [m.y]], compute_bounds_method=compute_fbbt_bounds)