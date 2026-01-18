import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def convert_pyomo2casadi(expr):
    """Convert a Pyomo expression tree to Casadi.

    This function replaces a Pyomo expression with a CasADi expression.
    This assumes that the `substitute_pyomo2casadi` function has
    been called, so the Pyomo expression contains CasADi variables
    and intrinsic functions.  The resulting expression can be used
    with the CasADi integrator.

    Args:
        expr: a Pyomo expression with CasADi variables and intrinsic
            functions

    Returns:
        a CasADi expression tree.
    """
    if not casadi_available:
        raise DAE_Error('CASADI is not installed.  Cannot convert a Pyomo expression to a Casadi expression.')
    visitor = Convert_Pyomo2Casadi_Visitor()
    return visitor.dfs_postorder_stack(expr)