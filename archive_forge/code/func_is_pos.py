from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
import cvxpy.utilities.linalg as eig_util
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import performance_utils as perf
def is_pos(self) -> bool:
    """Returns whether the constant is elementwise positive.
        """
    if self._cached_is_pos is None:
        if sp.issparse(self._value):
            self._cached_is_pos = False
        else:
            self._cached_is_pos = np.all(self._value > 0)
    return self._cached_is_pos