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
def test_maximize_obj(self):
    """Test the outer approximation decomposition algorithm."""
    with SolverFactory('mindtpy') as opt:
        model = ProposalModel().clone()
        model.objective.sense = maximize
        opt.solve(model, strategy='OA', mip_solver=required_solvers[1], nlp_solver=required_solvers[0])
        self.assertAlmostEqual(value(model.objective.expr), 14.83, places=1)