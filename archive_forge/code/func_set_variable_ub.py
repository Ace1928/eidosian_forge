from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_variable_ub(self, variable_id: int, ub: float) -> None:
    self._check_variable_id(variable_id)
    if ub == self.variables[variable_id].upper_bound:
        return
    self.variables[variable_id].upper_bound = ub
    for watcher in self._update_trackers:
        if variable_id < watcher.variables_checkpoint:
            watcher.variable_ubs.add(variable_id)