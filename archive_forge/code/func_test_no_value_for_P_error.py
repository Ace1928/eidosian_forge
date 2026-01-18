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
def test_no_value_for_P_error(self):
    m = models.makeBetweenStepsPaperExample()
    with self.assertRaisesRegex(GDP_Error, 'No value for P was given for disjunction disjunction! Please specify a value of P \\(number of partitions\\), if you do not specify the partitions directly.'):
        TransformationFactory('gdp.partition_disjuncts').apply_to(m)