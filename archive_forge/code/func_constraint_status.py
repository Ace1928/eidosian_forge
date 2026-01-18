import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def constraint_status(self, linear_constraints=None):
    """The constraint basis status associated to the best solution.

        If there is at least one primal feasible solution, this corresponds to the
        basis associated to the best primal feasible solution. An error will
        be raised if the best solution does not have an associated basis.


        Args:
          linear_constraints: an optional LinearConstraint or iterator of
            LinearConstraint indicating what constraint statuses to return. If not
            provided, returns a dictionary with the constraint statuses for all
            linear constraints.

        Returns:
          The constraint basis status associated to the best solution.

        Raises:
          ValueError: The best solution does not have an associated basis.
          TypeError: Argument is not None, a LinearConstraint or an iterable of
            LinearConstraint.
          KeyError: LinearConstraint values requested for an invalid
            linear constraint (e.g. is not a LinearConstraint or is a linear
            constraint for another model).
        """
    if not self.has_basis():
        raise ValueError(_NO_BASIS_ERROR)
    assert self.solutions[0].basis is not None
    if linear_constraints is None:
        return self.solutions[0].basis.constraint_status
    if isinstance(linear_constraints, model.LinearConstraint):
        return self.solutions[0].basis.constraint_status[linear_constraints]
    if isinstance(linear_constraints, Iterable):
        return [self.solutions[0].basis.constraint_status[c] for c in linear_constraints]
    raise TypeError(f'unsupported type in argument for constraint_status: {type(linear_constraints).__name__!r}')