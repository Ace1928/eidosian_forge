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
def get_coefficient(self, variable: Variable) -> float:
    self.model.check_compatible(variable)
    return self.model.storage.get_linear_constraint_coefficient(self._id, variable.id)