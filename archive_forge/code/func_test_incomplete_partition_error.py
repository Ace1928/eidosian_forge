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
def test_incomplete_partition_error(self):
    m = models.makeBetweenStepsPaperExample()
    self.assertRaisesRegex(GDP_Error, "Partition specified for disjunction containing Disjunct 'disjunction_disjuncts\\[0\\]' does not include all the variables that appear in the disjunction. The following variables are not assigned to any part of the partition: 'x\\[3\\]', 'x\\[4\\]'", TransformationFactory('gdp.partition_disjuncts').apply_to, m, variable_partitions=[[m.x[1]], [m.x[2]]], compute_bounds_method=compute_fbbt_bounds)