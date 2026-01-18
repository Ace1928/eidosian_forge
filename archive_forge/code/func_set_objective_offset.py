from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_objective_offset(self, offset: float) -> None:
    if self._objective_offset == offset:
        return
    self._objective_offset = offset
    for watcher in self._update_trackers:
        watcher.objective_offset = True