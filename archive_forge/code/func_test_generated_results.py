from io import StringIO
from typing import Sequence, Dict, Optional, Mapping, MutableMapping
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver import results
from pyomo.contrib.solver import solution
import pyomo.environ as pyo
from pyomo.core.base.var import Var
def test_generated_results(self):
    m = pyo.ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.c1 = pyo.Constraint(expr=m.x == 1)
    m.c2 = pyo.Constraint(expr=m.y == 2)
    primals = {}
    primals[id(m.x)] = (m.x, 1)
    primals[id(m.y)] = (m.y, 2)
    duals = {}
    duals[m.c1] = 3
    duals[m.c2] = 4
    rc = {}
    rc[id(m.x)] = (m.x, 5)
    rc[id(m.y)] = (m.y, 6)
    res = results.Results()
    res.solution_loader = SolutionLoaderExample(primals=primals, duals=duals, reduced_costs=rc)
    res.solution_loader.load_vars()
    self.assertAlmostEqual(m.x.value, 1)
    self.assertAlmostEqual(m.y.value, 2)
    m.x.value = None
    m.y.value = None
    res.solution_loader.load_vars([m.y])
    self.assertIsNone(m.x.value)
    self.assertAlmostEqual(m.y.value, 2)
    duals2 = res.solution_loader.get_duals()
    self.assertAlmostEqual(duals[m.c1], duals2[m.c1])
    self.assertAlmostEqual(duals[m.c2], duals2[m.c2])
    duals2 = res.solution_loader.get_duals([m.c2])
    self.assertNotIn(m.c1, duals2)
    self.assertAlmostEqual(duals[m.c2], duals2[m.c2])
    rc2 = res.solution_loader.get_reduced_costs()
    self.assertAlmostEqual(rc[id(m.x)][1], rc2[m.x])
    self.assertAlmostEqual(rc[id(m.y)][1], rc2[m.y])
    rc2 = res.solution_loader.get_reduced_costs([m.y])
    self.assertNotIn(m.x, rc2)
    self.assertAlmostEqual(rc[id(m.y)][1], rc2[m.y])