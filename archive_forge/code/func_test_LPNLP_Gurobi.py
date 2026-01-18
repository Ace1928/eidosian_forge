import sys
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
@unittest.skipUnless('gurobi_persistent' in available_mip_solvers, 'gurobi_persistent solver is not available')
def test_LPNLP_Gurobi(self):
    """Test the LP/NLP decomposition algorithm."""
    with SolverFactory('mindtpy') as opt:
        for model in model_list:
            model = model.clone()
            results = opt.solve(model, strategy='OA', mip_solver='gurobi_persistent', nlp_solver=required_nlp_solvers, single_tree=True)
            self.assertIn(results.solver.termination_condition, [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=1)
            self.check_optimal_solution(model)