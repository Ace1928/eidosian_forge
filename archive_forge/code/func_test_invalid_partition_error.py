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
def test_invalid_partition_error(self):
    m = models.makeNonQuadraticNonlinearGDP()
    self.assertRaisesRegex(GDP_Error, "Variables which appear in the expression \\(x\\[1\\]\\*\\*4 \\+ x\\[2\\]\\*\\*4\\)\\*\\*0.25 are in different partitions, but this expression doesn't appear additively separable. Please expand it if it is additively separable or, more likely, ensure that all the constraints in the disjunction are additively separable with respect to the specified partition. If you did not specify a partition, only a value of P, note that to automatically partition the variables, we assume all the expressions are additively separable.", TransformationFactory('gdp.partition_disjuncts').apply_to, m, variable_partitions=[[m.x[3], m.x[2]], [m.x[1], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)