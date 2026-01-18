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
def is_dpp(self, context='dcp') -> bool:
    """The expression is a disciplined parameterized expression.
        """
    if context.lower() == 'dcp':
        return self.is_dcp(dpp=True)
    elif context.lower() == 'dgp':
        return self.is_dgp(dpp=True)
    else:
        raise ValueError('Unsupported context ', context)