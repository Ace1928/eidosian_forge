from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.util import (
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from typing import Callable
from pyomo.common.gsl import find_GSL
from pyomo.opt.results import SolverResults
def test_check_optimal_termination_condition_legacy_interface(self):
    results = SolverResults()
    results.solver.status = SolverStatus.ok
    results.solver.termination_condition = LegacyTerminationCondition.optimal
    self.assertTrue(check_optimal_termination(results))
    results.solver.termination_condition = LegacyTerminationCondition.unknown
    self.assertFalse(check_optimal_termination(results))
    results.solver.termination_condition = SolverStatus.aborted
    self.assertFalse(check_optimal_termination(results))