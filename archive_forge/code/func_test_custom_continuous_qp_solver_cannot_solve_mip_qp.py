import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
def test_custom_continuous_qp_solver_cannot_solve_mip_qp(self) -> None:
    self.custom_conic_solver.MIP_CAPABLE = False
    with self.assertRaises(cp.error.SolverError):
        self.solve_example_mixed_integer_qp(solver=self.custom_qp_solver)