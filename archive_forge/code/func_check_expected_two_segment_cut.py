import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def check_expected_two_segment_cut(self, m):
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    self.assertEqual(len(cuts), 1)
    cut_expr = cuts[0].body
    m.x.fix(0)
    m.disj1.indicator_var.fix(True)
    m.disj2.indicator_var.fix(False)
    self.assertAlmostEqual(value(cut_expr), 0)
    m.x.fix(2)
    m.disj2.indicator_var.fix(True)
    m.disj1.indicator_var.fix(False)
    self.assertAlmostEqual(value(cut_expr), 0)