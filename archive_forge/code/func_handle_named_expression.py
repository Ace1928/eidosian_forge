import collections
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from pyomo.core.expr import (
from typing import List
from pyomo.common.collections import Sequence
from pyomo.common.errors import PyomoException
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import native_types
def handle_named_expression(node, pn: List, include_named_exprs=True):
    if include_named_exprs:
        pn.append((type(node), 1))
    return (node.expr,)