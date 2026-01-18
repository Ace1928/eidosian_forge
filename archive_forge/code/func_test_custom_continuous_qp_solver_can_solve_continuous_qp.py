import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
def test_custom_continuous_qp_solver_can_solve_continuous_qp(self) -> None:
    with self.assertRaises(CustomQPSolverCalled):
        self.solve_example_qp(solver=self.custom_qp_solver)