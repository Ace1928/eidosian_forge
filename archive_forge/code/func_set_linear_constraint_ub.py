from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def set_linear_constraint_ub(self, linear_constraint_id: int, ub: float) -> None:
    self._check_linear_constraint_id(linear_constraint_id)
    if ub == self.linear_constraints[linear_constraint_id].upper_bound:
        return
    self.linear_constraints[linear_constraint_id].upper_bound = ub
    for watcher in self._update_trackers:
        if linear_constraint_id < watcher.linear_constraints_checkpoint:
            watcher.linear_constraint_ubs.add(linear_constraint_id)