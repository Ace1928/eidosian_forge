import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
@unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
def test_cuts_are_correct_facets_fme(self):
    m = models.makeTwoTermDisj_boxes()
    TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None, zero_tolerance=0)
    facet_extreme_pts = [(0, 1, 1, 3), (0, 1, 2, 3), (1, 0, 3, 1), (1, 0, 4, 1)]
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    self.assertEqual(len(cuts), 1)
    cut = cuts[0]
    cut_expr = cut.body
    lower = cut.lower
    upper = cut.upper
    for pt in facet_extreme_pts:
        m.d[0].binary_indicator_var.fix(pt[0])
        m.d[1].binary_indicator_var.fix(pt[1])
        m.x.fix(pt[2])
        m.y.fix(pt[3])
        if lower is not None:
            self.assertEqual(value(lower), value(cut_expr))
        if upper is not None:
            self.assertEqual(value(upper), value(cut_expr))