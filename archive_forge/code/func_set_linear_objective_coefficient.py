from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_linear_objective_coefficient(self, variable_id: int, value: float) -> None:
    self._check_variable_id(variable_id)
    if value == self.linear_objective_coefficient.get(variable_id, 0.0):
        return
    if value == 0.0:
        self.linear_objective_coefficient.pop(variable_id, None)
    else:
        self.linear_objective_coefficient[variable_id] = value
    for watcher in self._update_trackers:
        if variable_id < watcher.variables_checkpoint:
            watcher.linear_objective_coefficients.add(variable_id)