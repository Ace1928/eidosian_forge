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
def check_threeTermDisj_IndexedConstraints(self, m, lb):
    transBlock = m._pyomo_gdp_hull_reformulation
    hull = TransformationFactory('gdp.hull')
    self.assertEqual(len(list(m.component_objects(Block, descend_into=False))), 1)
    self.assertEqual(len(list(m.component_objects(Disjunct))), 1)
    for i in [1, 2, 3]:
        relaxed = transBlock.relaxedDisjuncts[i - 1]
        self.assertEqual(len(list(relaxed.disaggregatedVars.component_objects(Var))), i)
        self.assertEqual(len(list(relaxed.disaggregatedVars.component_data_objects(Var))), i)
        self.assertEqual(len(list(relaxed.component_objects(Constraint))), 1 + i)
        if lb == 0:
            self.assertEqual(len(list(relaxed.component_data_objects(Constraint))), i + i)
        else:
            self.assertEqual(len(list(relaxed.component_data_objects(Constraint))), 2 * i + i)
        for j in range(1, i + 1):
            cons = hull.get_transformed_constraints(m.d[i].c[j])
            self.assertEqual(len(cons), 1)
            self.assertIs(cons[0].parent_block(), relaxed)
    self.assertEqual(len(list(transBlock.component_objects(Var, descend_into=False))), 1)
    self.assertEqual(len(list(transBlock.component_data_objects(Var, descend_into=False))), 2)
    self.assertEqual(len(list(transBlock.component_objects(Constraint, descend_into=False))), 3)
    if lb == 0:
        self.assertEqual(len(list(transBlock.component_data_objects(Constraint, descend_into=False))), 6)
    else:
        self.assertEqual(len(list(transBlock.component_data_objects(Constraint, descend_into=False))), 8)