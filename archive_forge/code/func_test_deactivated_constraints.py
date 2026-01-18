from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.tests.test_fbbt import FbbtTestBase
from pyomo.common.errors import InfeasibleConstraintException
import math
def test_deactivated_constraints(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.c1 = pe.Constraint(expr=m.x == 1)
    m.c2 = pe.Constraint(expr=m.y == m.x)
    it = appsi.fbbt.IntervalTightener()
    it.config.deactivate_satisfied_constraints = True
    it.perform_fbbt(m)
    self.assertFalse(m.c1.active)
    self.assertFalse(m.c2.active)
    m.c2.activate()
    m.x.setlb(0)
    m.x.setub(2)
    m.y.setlb(None)
    m.y.setub(None)
    it.perform_fbbt(m)
    self.assertTrue(m.c2.active)
    self.assertAlmostEqual(m.y.lb, 0)
    self.assertAlmostEqual(m.y.ub, 2)