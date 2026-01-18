from typing import List, Optional, Union
import numpy as np
from cvxpy.atoms import (
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.exponential import OpRelEntrConeQuad
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
def at_least_2D(expr: Expression):
    """Upcast 0D and 1D to 2D.
    """
    if expr.ndim < 2:
        return reshape(expr, (expr.size, 1))
    else:
        return expr