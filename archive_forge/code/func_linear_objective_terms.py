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
def linear_objective_terms(self) -> Iterator[LinearTerm]:
    """Yields variable coefficient pairs for variables with nonzero objective coefficient in undefined order."""
    for term in self.storage.get_linear_objective_coefficients():
        yield LinearTerm(variable=self._get_or_make_variable(term.variable_id), coefficient=term.coefficient)