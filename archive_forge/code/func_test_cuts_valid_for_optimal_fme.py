import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
@unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
def test_cuts_valid_for_optimal_fme(self):
    m = models.fourCircles()
    TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme)
    self.check_cuts_valid_for_optimal(m)