import os
from os.path import abspath, dirname
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet
from pyomo.core import (
from pyomo.core.base import TransformationFactory
from pyomo.core.expr import log
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.gdp import Disjunction, Disjunct
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import SolverFactory, check_available_solvers
import pyomo.contrib.fme.fourier_motzkin_elimination
from io import StringIO
import logging
import random
@unittest.skipIf(not 'glpk' in solvers, 'glpk not available')
def test_model_with_unrelated_nonlinear_expressions(self):
    m = ConcreteModel()
    m.x = Var([1, 2, 3], bounds=(0, 3))
    m.y = Var()
    m.z = Var()

    @m.Constraint([1, 2])
    def cons(m, i):
        return m.x[i] <= m.y ** i
    m.cons2 = Constraint(expr=m.x[1] >= m.y)
    m.cons3 = Constraint(expr=m.x[2] >= m.z - 3)
    m.cons4 = Constraint(expr=m.x[3] <= log(m.y + 1))
    fme = TransformationFactory('contrib.fourier_motzkin_elimination')
    fme.apply_to(m, vars_to_eliminate=m.x, projected_constraints_name='projected_constraints', constraint_filtering_callback=None)
    constraints = m.projected_constraints
    cons = constraints[5]
    self.assertEqual(value(cons.lower), 0)
    assertExpressionsEqual(self, cons.body, m.y)
    cons = constraints[6]
    self.assertEqual(value(cons.lower), -3)
    assertExpressionsEqual(self, cons.body, -m.y)
    cons = constraints[2]
    self.assertEqual(value(cons.lower), -3)
    assertExpressionsEqual(self, cons.body, -m.z + m.y ** 2)
    cons = constraints[4]
    self.assertEqual(cons.lower, -6)
    assertExpressionsEqual(self, cons.body, -m.z)
    cons = constraints[1]
    self.assertEqual(value(cons.lower), 0)
    assertExpressionsEqual(self, cons.body, log(m.y + 1))
    cons = constraints[3]
    self.assertEqual(value(cons.lower), 0)
    assertExpressionsEqual(self, cons.body, m.y ** 2)
    pts = [(1, 4), (3, 6), (3, 0), (0, 0), (2, 6)]
    for pt in pts:
        m.y.fix(pt[0])
        m.z.fix(pt[1])
        for i in constraints:
            self.assertLessEqual(value(constraints[i].lower), value(constraints[i].body))
    m.y.fixed = False
    m.z.fixed = False
    constraints[2].deactivate()
    constraints[3].deactivate()
    constraints[1].deactivate()
    fme.post_process_fme_constraints(m, SolverFactory('glpk'), projected_constraints=m.projected_constraints)
    self.assertEqual(len(constraints), 6)
    m.some_new_cons = Constraint(expr=m.y <= 2)
    fme.post_process_fme_constraints(m, SolverFactory('glpk'), projected_constraints=m.projected_constraints)
    self.assertEqual(len(constraints), 5)
    self.assertIsNone(dict(constraints).get(6))