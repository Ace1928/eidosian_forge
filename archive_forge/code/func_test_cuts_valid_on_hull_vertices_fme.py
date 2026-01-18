import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
@unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
def test_cuts_valid_on_hull_vertices_fme(self):
    m = models.makeTwoTermDisj_boxes()
    TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
    self.check_cuts_valid_on_hull_vertices(m, TOL=0)