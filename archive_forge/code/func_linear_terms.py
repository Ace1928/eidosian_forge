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
def linear_terms(self) -> Iterator[LinearTerm]:
    """Yields variable coefficient pairs for variables with nonzero objective coefficient in undefined order."""
    yield from self.model.linear_objective_terms()