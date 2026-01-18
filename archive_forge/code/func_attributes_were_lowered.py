from __future__ import annotations
from typing import Any, Iterable
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
def attributes_were_lowered(self) -> bool:
    """True iff variable generated when lowering a variable with attributes.
        """
    return self._variable_with_attributes is not None