from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.util import (
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from typing import Callable
from pyomo.common.gsl import find_GSL
from pyomo.opt.results import SolverResults
def test_get_objective_raise(self):
    model = self.simple_model()
    model.OBJ2 = pyo.Objective(expr=model.x[1] - 4 * model.x[2])
    with self.assertRaises(ValueError):
        get_objective(model)