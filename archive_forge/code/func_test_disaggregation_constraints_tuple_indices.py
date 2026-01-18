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
def test_disaggregation_constraints_tuple_indices(self):
    m = models.makeTwoTermMultiIndexedDisjunction()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    relaxedDisjuncts = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
    disaggregatedVars = {(1, 'A'): [hull.get_disaggregated_var(m.a[1, 'A'], m.disjunct[0, 1, 'A']), hull.get_disaggregated_var(m.a[1, 'A'], m.disjunct[1, 1, 'A'])], (1, 'B'): [hull.get_disaggregated_var(m.a[1, 'B'], m.disjunct[0, 1, 'B']), hull.get_disaggregated_var(m.a[1, 'B'], m.disjunct[1, 1, 'B'])], (2, 'A'): [hull.get_disaggregated_var(m.a[2, 'A'], m.disjunct[0, 2, 'A']), hull.get_disaggregated_var(m.a[2, 'A'], m.disjunct[1, 2, 'A'])], (2, 'B'): [hull.get_disaggregated_var(m.a[2, 'B'], m.disjunct[0, 2, 'B']), hull.get_disaggregated_var(m.a[2, 'B'], m.disjunct[1, 2, 'B'])]}
    for i, disVars in disaggregatedVars.items():
        cons = hull.get_disaggregation_constraint(m.a[i], m.disjunction[i])
        self.assertEqual(cons.lower, 0)
        self.assertEqual(cons.upper, 0)
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, m.a[i], 1)
        ct.check_linear_coef(self, repn, disVars[0], -1)
        self.assertTrue(disVars[1].is_fixed())
        self.assertEqual(value(disVars[1]), 0)