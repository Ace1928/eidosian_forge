from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_variable_is_integer(self, variable_id: int, is_integer: bool) -> None:
    self._check_variable_id(variable_id)
    if is_integer == self.variables[variable_id].is_integer:
        return
    self.variables[variable_id].is_integer = is_integer
    for watcher in self._update_trackers:
        if variable_id < watcher.variables_checkpoint:
            watcher.variable_integers.add(variable_id)