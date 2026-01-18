from typing import List, Optional, Union
import numpy as np
from cvxpy.atoms import (
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.exponential import OpRelEntrConeQuad
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
def matrix_frac_canon(expr, real_args: List[Union[Expression, None]], imag_args: List[Union[Expression, None]], real2imag):
    """Convert matrix_frac to real.
    """
    if real_args[0] is None:
        real_args[0] = np.zeros(imag_args[0].shape)
    if imag_args[0] is None:
        imag_args[0] = np.zeros(real_args[0].shape)
    vec = vstack([at_least_2D(real_args[0]), at_least_2D(imag_args[0])])
    if real_args[1] is None:
        real_args[1] = np.zeros(imag_args[1].shape)
    elif imag_args[1] is None:
        imag_args[1] = np.zeros(real_args[1].shape)
    matrix = bmat([[real_args[1], -imag_args[1]], [imag_args[1], real_args[1]]])
    return (expr.copy([vec, matrix]), None)