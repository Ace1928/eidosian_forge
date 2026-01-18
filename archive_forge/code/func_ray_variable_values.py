import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def ray_variable_values(self, variables=None):
    """The variable values from the first primal ray.

        An error will be raised if there are no primal rays.

        Args:
          variables: an optional Variable or iterator of Variables indicating what
            variable values to return. If not provided, variable_values() returns a
            dictionary with the variable values for all variables.

        Returns:
          The variable values from the first primal ray.

        Raises:
          ValueError: There are no primal rays.
          TypeError: Argument is not None, a Variable or an iterable of Variables.
          KeyError: Variable values requested for an invalid variable (e.g. is not a
            Variable or is a variable for another model).
        """
    if not self.has_ray():
        raise ValueError('No primal ray available.')
    if variables is None:
        return self.primal_rays[0].variable_values
    if isinstance(variables, model.Variable):
        return self.primal_rays[0].variable_values[variables]
    if isinstance(variables, Iterable):
        return [self.primal_rays[0].variable_values[v] for v in variables]
    raise TypeError(f'unsupported type in argument for ray_variable_values: {type(variables).__name__!r}')