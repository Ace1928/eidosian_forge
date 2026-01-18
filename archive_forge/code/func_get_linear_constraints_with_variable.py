from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_linear_constraints_with_variable(self, variable_id: int) -> Iterator[int]:
    self._check_variable_id(variable_id)
    yield from self.variables[variable_id].linear_constraint_nonzeros