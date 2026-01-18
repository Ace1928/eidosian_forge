import collections
import enum
import functools
import itertools
import logging
import operator
import sys
from pyomo.common.collections import Sequence, ComponentMap, ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import is_fixed, value
import pyomo.core.expr as EXPR
import pyomo.core.kernel as kernel
def apply_node_operation(node, args):
    try:
        ans = node._apply_operation(args)
        if ans != ans and ans.__class__ is not InvalidNumber:
            ans = InvalidNumber(ans, "Evaluating '{node}' returned NaN")
        return ans
    except:
        exc_msg = str(sys.exc_info()[1])
        logger.warning("Exception encountered evaluating expression '%s(%s)'\n\tmessage: %s\n\texpression: %s" % (node.name, ', '.join(map(str, args)), exc_msg, node))
        if HALT_ON_EVALUATION_ERROR:
            raise
        return InvalidNumber(nan, exc_msg)