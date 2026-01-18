from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_linear_constraint_coefficient(self, linear_constraint_id: int, variable_id: int, value: float) -> None:
    self._check_linear_constraint_id(linear_constraint_id)
    self._check_variable_id(variable_id)
    if value == self._linear_constraint_matrix.get((linear_constraint_id, variable_id), 0.0):
        return
    if value == 0.0:
        self._linear_constraint_matrix.pop((linear_constraint_id, variable_id), None)
        self.variables[variable_id].linear_constraint_nonzeros.discard(linear_constraint_id)
        self.linear_constraints[linear_constraint_id].variable_nonzeros.discard(variable_id)
    else:
        self._linear_constraint_matrix[linear_constraint_id, variable_id] = value
        self.variables[variable_id].linear_constraint_nonzeros.add(linear_constraint_id)
        self.linear_constraints[linear_constraint_id].variable_nonzeros.add(variable_id)
    for watcher in self._update_trackers:
        if variable_id < watcher.variables_checkpoint and linear_constraint_id < watcher.linear_constraints_checkpoint:
            watcher.linear_constraint_matrix.add((linear_constraint_id, variable_id))