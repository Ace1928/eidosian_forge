from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.util import (
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from typing import Callable
from pyomo.common.gsl import find_GSL
from pyomo.opt.results import SolverResults
def basics_helper(self, collector: Callable, *args):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.E = pyo.Expression(expr=2 * m.z + 1)
    m.y.fix(3)
    e = m.x * m.y + m.x * m.E
    named_exprs, var_list, fixed_vars, external_funcs = collector(e, *args)
    self.assertEqual([m.E], named_exprs)
    self.assertEqual([m.x, m.y, m.z], var_list)
    self.assertEqual([m.y], fixed_vars)
    self.assertEqual([], external_funcs)