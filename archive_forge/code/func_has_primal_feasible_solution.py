import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def has_primal_feasible_solution(self) -> bool:
    """Indicates if at least one primal feasible solution is available.

        When termination.reason is TerminationReason.OPTIMAL or
        TerminationReason.FEASIBLE, this is guaranteed to be true and need not be
        checked.

        Returns:
          True if there is at least one primal feasible solution is available,
          False, otherwise.
        """
    if not self.solutions:
        return False
    return self.solutions[0].primal_solution is not None and self.solutions[0].primal_solution.feasibility_status == solution.SolutionStatus.FEASIBLE