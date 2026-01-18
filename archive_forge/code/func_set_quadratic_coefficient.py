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
def set_quadratic_coefficient(self, first_variable: Variable, second_variable: Variable, coef: float) -> None:
    self.model.check_compatible(first_variable)
    self.model.check_compatible(second_variable)
    self.model.storage.set_quadratic_objective_coefficient(first_variable.id, second_variable.id, coef)