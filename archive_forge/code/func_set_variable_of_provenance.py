from __future__ import annotations
from typing import Any, Iterable
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
def set_variable_of_provenance(self, variable: Variable) -> None:
    assert variable.attributes
    self._variable_with_attributes = variable