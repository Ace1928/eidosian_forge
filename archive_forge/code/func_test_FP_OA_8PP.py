import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.gdpopt.util import is_feasible
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.mindtpy.tests.feasibility_pump1 import FeasPump1
from pyomo.contrib.mindtpy.tests.feasibility_pump2 import FeasPump2
def test_FP_OA_8PP(self):
    """Test the FP-OA algorithm."""
    with SolverFactory('mindtpy') as opt:
        for model in model_list:
            model = model.clone()
            results = opt.solve(model, strategy='OA', init_strategy='FP', mip_solver=required_solvers[1], nlp_solver=required_solvers[0])
            self.assertIn(results.solver.termination_condition, [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.objective.expr), model.optimal_value, places=1)
            self.check_optimal_solution(model)