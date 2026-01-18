from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_quadratic_objective_adjacent_variables(self, variable_id: int) -> Iterator[int]:
    self._check_variable_id(variable_id)
    yield from self._quadratic_objective_coefficients.get_adjacent_variables(variable_id)