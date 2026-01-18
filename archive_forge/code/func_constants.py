from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
import cvxpy.utilities.linalg as eig_util
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import performance_utils as perf
def constants(self) -> List['Constant']:
    """Returns self as a constant.
        """
    return [self]