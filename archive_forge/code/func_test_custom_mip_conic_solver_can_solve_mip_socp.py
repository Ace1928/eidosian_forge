import unittest
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
def test_custom_mip_conic_solver_can_solve_mip_socp(self) -> None:
    self.custom_conic_solver.MIP_CAPABLE = True
    supported_constraints = self.custom_conic_solver.SUPPORTED_CONSTRAINTS
    self.custom_conic_solver.MI_SUPPORTED_CONSTRAINTS = supported_constraints
    with self.assertRaises(CustomConicSolverCalled):
        self.solve_example_mixed_integer_socp(solver=self.custom_conic_solver)