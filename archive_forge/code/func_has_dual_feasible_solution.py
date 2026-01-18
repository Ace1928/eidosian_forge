import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def has_dual_feasible_solution(self) -> bool:
    """Indicates if the best solution has an associated dual feasible solution.

        This is NOT guaranteed to be true when termination.reason is
        TerminationReason.Optimal. It also may be true even when the best solution
        does not have an associated primal feasible solution.

        Returns:
          True if the best solution has an associated dual feasible solution.
        """
    if not self.solutions:
        return False
    return self.solutions[0].dual_solution is not None and self.solutions[0].dual_solution.feasibility_status == solution.SolutionStatus.FEASIBLE