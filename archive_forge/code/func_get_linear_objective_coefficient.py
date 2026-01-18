from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_linear_objective_coefficient(self, variable_id: int) -> float:
    self._check_variable_id(variable_id)
    return self.linear_objective_coefficient.get(variable_id, 0.0)