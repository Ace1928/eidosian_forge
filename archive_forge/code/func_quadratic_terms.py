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
def quadratic_terms(self) -> Iterator[QuadraticTerm]:
    """Yields quadratic terms with nonzero objective coefficient in undefined order."""
    yield from self.model.quadratic_objective_terms()