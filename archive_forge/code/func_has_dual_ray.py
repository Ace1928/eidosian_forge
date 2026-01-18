import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def has_dual_ray(self) -> bool:
    """Indicates if at least one dual ray is available.

        This is NOT guaranteed to be true when termination.reason is
        TerminationReason.Infeasible.

        Returns:
          True if at least one dual ray is available.
        """
    return bool(self.dual_rays)