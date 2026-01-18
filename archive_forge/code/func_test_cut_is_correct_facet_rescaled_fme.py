import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
@unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
def test_cut_is_correct_facet_rescaled_fme(self):
    m = models.to_break_constraint_tolerances()
    TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
    cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
    self.assertEqual(len(cuts), 1)
    cut_extreme_points = [(1, 0, 2, 127), (0, 1, 120, 3)]
    for pt in cut_extreme_points:
        m.x.fix(pt[2])
        m.y.fix(pt[3])
        m.disjunct1.binary_indicator_var.fix(pt[0])
        m.disjunct2.binary_indicator_var.fix(pt[1])
        self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body))
        self.assertLessEqual(value(cuts[0].lower), value(cuts[0].body))