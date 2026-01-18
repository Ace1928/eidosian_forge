from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def get_linear_constraint_matrix_entries(self) -> Iterator[model_storage.LinearConstraintMatrixIdEntry]:
    for (constraint, variable), coef in self._linear_constraint_matrix.items():
        yield model_storage.LinearConstraintMatrixIdEntry(linear_constraint_id=constraint, variable_id=variable, coefficient=coef)