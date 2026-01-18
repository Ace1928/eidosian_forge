from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.util import (
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from typing import Callable
from pyomo.common.gsl import find_GSL
from pyomo.opt.results import SolverResults
def external_func_helper(self, collector: Callable, *args):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find amplgsl.dll library')
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.hypot = pyo.ExternalFunction(library=DLL, function='gsl_hypot')
    func = m.hypot(m.x, m.x * m.y)
    m.E = pyo.Expression(expr=2 * func)
    m.y.fix(3)
    e = m.z + m.x * m.E
    named_exprs, var_list, fixed_vars, external_funcs = collector(e, *args)
    self.assertEqual([m.E], named_exprs)
    self.assertEqual([m.z, m.x, m.y], var_list)
    self.assertEqual([m.y], fixed_vars)
    self.assertEqual([func], external_funcs)