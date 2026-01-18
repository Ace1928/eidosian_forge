from typing import List, Optional, Union
import numpy as np
from cvxpy.atoms import (
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.exponential import OpRelEntrConeQuad
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
def hermitian_canon(expr: Expression, real_args: List[Union[Expression, None]], imag_args: List[Union[Expression, None]], real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    assert len(real_args) == 1 and len(imag_args) == 1
    expr_canon = expand_and_reapply(expr, real_args[0], imag_args[0])
    return (expr_canon, None)