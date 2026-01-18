import abc
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import interface as intf
from cvxpy import utilities as u
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import performance_utils as perf
from cvxpy.utilities.deterministic import unique_list
def new_numeric(self, values):
    interface = intf.DEFAULT_INTF
    values = [interface.const_to_matrix(v, convert_scalars=True) for v in values]
    result = numeric_func(self, values)
    return intf.DEFAULT_INTF.const_to_matrix(result)