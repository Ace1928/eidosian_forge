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
def add_quadratic(self, objective: QuadraticTypes) -> None:
    """Adds the provided quadratic expression `objective` to the objective function."""
    if not isinstance(objective, (QuadraticBase, LinearBase, int, float)):
        raise TypeError(f'unsupported type in objective argument for Objective.add(): {type(objective).__name__!r}')
    objective_expr = as_flat_quadratic_expression(objective)
    self.offset += objective_expr.offset
    for var, coefficient in objective_expr.linear_terms.items():
        self.set_linear_coefficient(var, self.get_linear_coefficient(var) + coefficient)
    for key, coefficient in objective_expr.quadratic_terms.items():
        self.set_quadratic_coefficient(key.first_var, key.second_var, self.get_quadratic_coefficient(key.first_var, key.second_var) + coefficient)