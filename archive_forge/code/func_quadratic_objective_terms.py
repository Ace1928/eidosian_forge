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
def quadratic_objective_terms(self) -> Iterator[QuadraticTerm]:
    """Yields the quadratic terms with nonzero objective coefficient in undefined order."""
    for term in self.storage.get_quadratic_objective_coefficients():
        var1 = self._get_or_make_variable(term.id_key.id1)
        var2 = self._get_or_make_variable(term.id_key.id2)
        yield QuadraticTerm(key=QuadraticTermKey(var1, var2), coefficient=term.coefficient)