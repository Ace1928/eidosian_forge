import dataclasses
import datetime
import enum
import math
from typing import Dict, List, Mapping, Optional, Set, Union
from ortools.math_opt import callback_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def add_user_cut(self, bounded_expr: Optional[Union[bool, model.BoundedLinearTypes]]=None, *, lb: Optional[float]=None, ub: Optional[float]=None, expr: Optional[model.LinearTypes]=None) -> None:
    """Shortcut for add_generated_constraint(..., is_lazy=False)."""
    self.add_generated_constraint(bounded_expr, lb=lb, ub=ub, expr=expr, is_lazy=False)