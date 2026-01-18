import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def bounded(self) -> bool:
    """Returns true only if the problem has been shown to be feasible and bounded."""
    return self.termination.problem_status.primal_status == FeasibilityStatus.FEASIBLE and self.termination.problem_status.dual_status == FeasibilityStatus.FEASIBLE