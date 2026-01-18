import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    if arg1[0] is _CONSTANT and arg2[0] is _CONSTANT and (arg3[0] is _CONSTANT):
        return (_CONSTANT, node._apply_operation((arg1[1], arg2[1], arg3[1])))
    op = visitor.template.strict_inequality_map[node.strict]
    nl, args, named = node_result_to_amplrepn(arg1).compile_repn(visitor, visitor.template.and_expr + op[0])
    nl2, args2, named = node_result_to_amplrepn(arg2).compile_repn(visitor, '', None, named)
    nl += nl2 + op[1] + nl2
    args.extend(args2)
    args.extend(args2)
    nonlin = node_result_to_amplrepn(arg3).compile_repn(visitor, nl, args, named)
    return (_GENERAL, AMPLRepn(0, None, nonlin))