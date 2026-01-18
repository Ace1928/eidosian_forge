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
def row_nonzeros(self, linear_constraint: LinearConstraint) -> Iterator[Variable]:
    """Yields the variables with nonzero coefficient for this linear constraint."""
    for var_id in self.storage.get_variables_for_linear_constraint(linear_constraint.id):
        yield self._get_or_make_variable(var_id)