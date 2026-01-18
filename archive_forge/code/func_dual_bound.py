import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def dual_bound(self) -> float:
    """Returns a dual bound on the optimal objective value as described in ObjectiveBounds.

        Will return a valid (possibly infinite) bound even if no dual feasible
        solutions are available.
        """
    return self.termination.objective_bounds.dual_bound