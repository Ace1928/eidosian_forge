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
def delete_linear_constraint(self, linear_constraint: LinearConstraint) -> None:
    self.check_compatible(linear_constraint)
    self.storage.delete_linear_constraint(linear_constraint.id)
    del self._linear_constraint_ids[linear_constraint.id]