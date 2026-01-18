import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def ray_reduced_costs(self, variables=None):
    """The reduced costs from the first dual ray.

        An error will be raised if there are no dual rays.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            reduced costs to return. If not provided, ray_reduced_costs() returns a
            dictionary with the reduced costs for all variables.

        Returns:
          The reduced costs from the first dual ray.

        Raises:
          ValueError: There are no dual rays.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
    if not self.has_dual_ray():
        raise ValueError('No dual ray available.')
    if variables is None:
        return self.dual_rays[0].reduced_costs
    if isinstance(variables, model.Variable):
        return self.dual_rays[0].reduced_costs[variables]
    if isinstance(variables, Iterable):
        return [self.dual_rays[0].reduced_costs[v] for v in variables]
    raise TypeError(f'unsupported type in argument for ray_reduced_costs: {type(variables).__name__!r}')