from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_quadratic_objective_coefficient(self, first_variable_id: int, second_variable_id: int) -> float:
    self._check_variable_id(first_variable_id)
    self._check_variable_id(second_variable_id)
    return self._quadratic_objective_coefficients.get_coefficient(first_variable_id, second_variable_id)