import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def ray_dual_values(self, linear_constraints=None):
    """The dual values from the first dual ray.

        An error will be raised if there are no dual rays.

        Args:
          linear_constraints: an optional LinearConstraint or iterator of
            LinearConstraint indicating what dual values to return. If not provided,
            ray_dual_values() returns a dictionary with the dual values for all
            linear constraints.

        Returns:
          The dual values from the first dual ray.

        Raises:
          ValueError: There are no dual rays.
          TypeError: Argument is not None, a LinearConstraint or an iterable of
            LinearConstraint.
          KeyError: LinearConstraint values requested for an invalid
            linear constraint (e.g. is not a LinearConstraint or is a linear
            constraint for another model).
        """
    if not self.has_dual_ray():
        raise ValueError('No dual ray available.')
    if linear_constraints is None:
        return self.dual_rays[0].dual_values
    if isinstance(linear_constraints, model.LinearConstraint):
        return self.dual_rays[0].dual_values[linear_constraints]
    if isinstance(linear_constraints, Iterable):
        return [self.dual_rays[0].dual_values[v] for v in linear_constraints]
    raise TypeError(f'unsupported type in argument for ray_dual_values: {type(linear_constraints).__name__!r}')