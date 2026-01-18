from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.MINLP4_simple import SimpleMINLP4
from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value, maximize
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.opt import TerminationCondition
def test_OA_partition_obj_nonlinear_terms(self):
    """Test the outer approximation decomposition algorithm (partition_obj_nonlinear_terms)."""
    with SolverFactory('mindtpy') as opt:
        for model in obj_nonlinear_sum_model_list:
            model = model.clone()
            results = opt.solve(model, strategy='OA', mip_solver=required_solvers[1], nlp_solver=required_solvers[0], partition_obj_nonlinear_terms=True)
            self.assertIn(results.solver.termination_condition, [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=1)
            self.check_optimal_solution(model)