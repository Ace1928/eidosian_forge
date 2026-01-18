from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def test_mapping_method_errors(self):
    m = models.makeTwoTermDisj_Nonlinear()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    with self.assertRaisesRegex(GDP_Error, ".*Either 'w' is not a disaggregated variable, or the disjunction that disaggregates it has not been properly transformed."):
        hull.get_var_bounds_constraint(m.w)
    log = StringIO()
    with LoggingIntercept(log, 'pyomo.gdp.hull', logging.ERROR):
        self.assertRaisesRegex(KeyError, '.*_pyomo_gdp_hull_reformulation.relaxedDisjuncts\\[1\\].disaggregatedVars.w', hull.get_disaggregation_constraint, m.d[1].transformation_block.disaggregatedVars.w, m.disjunction)
    self.assertRegex(log.getvalue(), ".*It doesn't appear that '_pyomo_gdp_hull_reformulation.relaxedDisjuncts\\[1\\].disaggregatedVars.w' is a variable that was disaggregated by Disjunction 'disjunction'")
    with self.assertRaisesRegex(GDP_Error, ".*'w' does not appear to be a disaggregated variable"):
        hull.get_src_var(m.w)
    with self.assertRaisesRegex(GDP_Error, ".*It does not appear '_pyomo_gdp_hull_reformulation.relaxedDisjuncts\\[1\\].disaggregatedVars.w' is a variable that appears in disjunct 'd\\[1\\]'"):
        hull.get_disaggregated_var(m.d[1].transformation_block.disaggregatedVars.w, m.d[1])
    m.random_disjunction = Disjunction(expr=[m.w == 2, m.w >= 7])
    self.assertRaisesRegex(GDP_Error, "Disjunction 'random_disjunction' has not been properly transformed: None of its disjuncts are transformed.", hull.get_disaggregation_constraint, m.w, m.random_disjunction)
    self.assertRaisesRegex(GDP_Error, "Disjunct 'random_disjunction_disjuncts\\[0\\]' has not been transformed", hull.get_disaggregated_var, m.w, m.random_disjunction.disjuncts[0])