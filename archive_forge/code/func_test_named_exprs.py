from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.tests.test_fbbt import FbbtTestBase
from pyomo.common.errors import InfeasibleConstraintException
import math
def test_named_exprs(self):
    m = pe.ConcreteModel()
    m.a = pe.Set(initialize=[1, 2, 3])
    m.x = pe.Var(m.a, bounds=(0, None))
    m.e = pe.Expression(m.a)
    for i in m.a:
        m.e[i].expr = i * m.x[i]
    m.c = pe.Constraint(expr=sum(m.e.values()) == 0)
    it = appsi.fbbt.IntervalTightener()
    it.perform_fbbt(m)
    for x in m.x.values():
        self.assertAlmostEqual(x.lb, 0)
        self.assertAlmostEqual(x.ub, 0)