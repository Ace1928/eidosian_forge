import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def set_quadratic_objective(self, objective: QuadraticTypes, *, is_maximize: bool) -> None:
    """Sets the objective to optimize the provided linear expression `objective`."""
    if not isinstance(objective, (QuadraticBase, LinearBase, int, float)):
        raise TypeError(f'unsupported type in objective argument for set_quadratic_objective: {type(objective).__name__!r}')
    self.storage.clear_objective()
    self._objective.is_maximize = is_maximize
    objective_expr = as_flat_quadratic_expression(objective)
    self._objective.offset = objective_expr.offset
    for var, coefficient in objective_expr.linear_terms.items():
        self._objective.set_linear_coefficient(var, coefficient)
    for key, coefficient in objective_expr.quadratic_terms.items():
        self._objective.set_quadratic_coefficient(key.first_var, key.second_var, coefficient)