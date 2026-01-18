import abc
import warnings
from functools import wraps
from typing import Tuple
import numpy as np
import cvxpy.settings as s
import cvxpy.utilities as u
import cvxpy.utilities.key_utils as ku
import cvxpy.utilities.performance_utils as perf
from cvxpy import error
from cvxpy.constraints import PSD, Equality, Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
from cvxpy.utilities.shape import size_from_shape
def is_log_log_constant(self) -> bool:
    """Is the expression log-log constant, ie, elementwise positive?
        """
    if not self.is_constant():
        return False
    if isinstance(self, (cvxtypes.constant(), cvxtypes.parameter())):
        return self.is_pos()
    else:
        return self.value is not None and np.all(self.value > 0)