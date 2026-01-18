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
def is_matrix(self) -> bool:
    """Is the expression a matrix?
        """
    return self.ndim == 2 and self.shape[0] > 1 and (self.shape[1] > 1)