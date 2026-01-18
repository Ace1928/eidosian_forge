import math
import copy
import re
import io
import pyomo.environ as pyo
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr import (
from pyomo.core.expr.visitor import identify_components
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.expr.template_expr import (
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.core.base.param import _ParamData, ScalarParam, IndexedParam
from pyomo.core.base.set import _SetData
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from pyomo.common.collections.component_map import ComponentMap
from pyomo.common.collections.component_set import ComponentSet
from pyomo.core.expr.template_expr import (
from pyomo.core.expr.numeric_expr import NPV_SumExpression, NPV_DivisionExpression
from pyomo.core.base.block import IndexedBlock
from pyomo.core.base.external import _PythonCallbackFunctionID
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.block import _BlockData
from pyomo.repn.util import ExprType
from pyomo.common import DeveloperError
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.dependencies import numpy as np, numpy_available
def precedenceChecker(node, arg1, arg2=None):
    childPrecedence = []
    for a in node.args:
        if hasattr(a, 'PRECEDENCE'):
            if a.PRECEDENCE is None:
                childPrecedence.append(-1)
            else:
                childPrecedence.append(a.PRECEDENCE)
        else:
            childPrecedence.append(-1)
    if hasattr(node, 'PRECEDENCE'):
        precedence = node.PRECEDENCE
    else:
        raise DeveloperError('This error should never be thrown, node does not have a precedence.  Report to developers')
    if childPrecedence[0] > precedence:
        arg1 = ' \\left( ' + arg1 + ' \\right) '
    if arg2 is not None:
        if childPrecedence[1] > precedence:
            arg2 = ' \\left( ' + arg2 + ' \\right) '
    return (arg1, arg2)