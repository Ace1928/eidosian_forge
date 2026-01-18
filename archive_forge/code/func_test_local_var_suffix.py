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
def test_local_var_suffix(self):
    hull = TransformationFactory('gdp.hull')
    model = ConcreteModel()
    model.x = Var(bounds=(5, 100))
    model.y = Var(bounds=(0, 100))
    model.d1 = Disjunct()
    model.d1.c = Constraint(expr=model.y >= model.x)
    model.d2 = Disjunct()
    model.d2.z = Var(bounds=(-9, -7))
    model.d2.c = Constraint(expr=model.y >= model.d2.z)
    model.disj = Disjunction(expr=[model.d1, model.d2])
    m = hull.create_using(model)
    self.assertEqual(m.d2.z.lb, -9)
    self.assertEqual(m.d2.z.ub, -7)
    z_disaggregated = m.d2.transformation_block.disaggregatedVars.component('d2.z')
    self.assertIsInstance(z_disaggregated, Var)
    self.assertIs(z_disaggregated, hull.get_disaggregated_var(m.d2.z, m.d2))
    model.d2.LocalVars = Suffix(direction=Suffix.LOCAL)
    model.d2.LocalVars[model.d2] = [model.d2.z]
    m = hull.create_using(model)
    self.assertEqual(m.d2.z.lb, -9)
    self.assertEqual(m.d2.z.ub, 0)
    self.assertIs(hull.get_disaggregated_var(m.d2.z, m.d2), m.d2.z)
    self.assertIsNone(m.d2.transformation_block.disaggregatedVars.component('z'))