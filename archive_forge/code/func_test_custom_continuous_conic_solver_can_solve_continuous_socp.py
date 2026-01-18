import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
def test_custom_continuous_conic_solver_can_solve_continuous_socp(self) -> None:
    with self.assertRaises(CustomConicSolverCalled):
        self.solve_example_socp(solver=self.custom_conic_solver)