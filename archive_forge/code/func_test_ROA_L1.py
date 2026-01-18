import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
def test_ROA_L1(self):
    """Test the LP/NLP decomposition algorithm."""
    with SolverFactory('mindtpy') as opt:
        for model in model_list:
            model = model.clone()
            results = opt.solve(model, strategy='OA', mip_solver=required_solvers[1], nlp_solver=required_solvers[0], add_regularization='level_L1')
            self.assertIn(results.solver.termination_condition, [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=1)
            self.check_optimal_solution(model)