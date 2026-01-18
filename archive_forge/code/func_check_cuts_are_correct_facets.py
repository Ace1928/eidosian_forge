import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
def check_cuts_are_correct_facets(self, m):
    cut1_tight_pts = [(1, 0, 3, 1), (0, 1, 1, 3)]
    facet2_extreme_pts = [(1, 0, 3, 1), (1, 0, 4, 1)]
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    self.assertEqual(len(cuts), 2)
    cut = cuts[0]
    cut_expr = cut.body
    lower = cut.lower
    upper = cut.upper
    for pt in cut1_tight_pts:
        m.d[0].binary_indicator_var.fix(pt[0])
        m.d[1].binary_indicator_var.fix(pt[1])
        m.x.fix(pt[2])
        m.y.fix(pt[3])
        if lower is not None:
            self.assertAlmostEqual(value(lower), value(cut_expr), places=6)
        if upper is not None:
            self.assertAlmostEqual(value(upper), value(cut_expr))
    cut = cuts[1]
    cut_expr = cut.body
    lower = cut.lower
    upper = cut.upper
    for pt in facet2_extreme_pts:
        m.d[0].binary_indicator_var.fix(pt[0])
        m.d[1].binary_indicator_var.fix(pt[1])
        m.x.fix(pt[2])
        m.y.fix(pt[3])
        if lower is not None:
            self.assertAlmostEqual(value(lower), value(cut_expr))
        if upper is not None:
            self.assertAlmostEqual(value(upper), value(cut_expr))