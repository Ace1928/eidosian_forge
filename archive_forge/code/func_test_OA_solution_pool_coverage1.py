from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.environ import SolverFactory, value, maximize
from pyomo.opt import TerminationCondition
@unittest.skipIf(not (ipopt_available and cplex_persistent_available), 'Required subsolvers are not available')
def test_OA_solution_pool_coverage1(self):
    """Test the outer approximation decomposition algorithm."""
    with SolverFactory('mindtpy') as opt:
        for model in model_list:
            model = model.clone()
            results = opt.solve(model, strategy='OA', init_strategy='rNLP', solution_pool=True, mip_solver='glpk', nlp_solver=required_solvers[0], num_solution_iteration=1)
            self.assertIn(results.solver.termination_condition, [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=2)
            self.check_optimal_solution(model)