import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def check_cut_is_correct_facet(self, m):
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    self.assertEqual(len(cuts), 2)
    cut1_tight_points = [(1, 0, 2, 10), (0, 1, 10, 3)]
    cut2_tight_points = [(1, 0, 2, 10), (1, 0, 0, 10)]
    for pt in cut1_tight_points:
        m.x.fix(pt[2])
        m.y.fix(pt[3])
        m.disjunct1.binary_indicator_var.fix(pt[0])
        m.disjunct2.binary_indicator_var.fix(pt[1])
        self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body), places=6)
    for pt in cut2_tight_points:
        m.x.fix(pt[2])
        m.y.fix(pt[3])
        m.disjunct1.binary_indicator_var.fix(pt[0])
        m.disjunct2.binary_indicator_var.fix(pt[1])
        self.assertAlmostEqual(value(cuts[1].lower), value(cuts[1].body), places=6)