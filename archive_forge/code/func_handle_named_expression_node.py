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
def handle_named_expression_node(visitor, node, arg1):
    _id = id(node)
    repn = node_result_to_amplrepn(arg1)
    expression_source = [None, None, False]
    visitor.subexpression_cache[_id] = (node, repn, expression_source)
    if not visitor.use_named_exprs:
        return (_GENERAL, repn.duplicate())
    mult, repn.mult = (repn.mult, 1)
    if repn.named_exprs is None:
        repn.named_exprs = set()
    repn.nl = (visitor.template.var, (_id,))
    if repn.nonlinear:
        if repn.nonlinear.__class__ is list:
            repn.compile_nonlinear_fragment(visitor)
        if repn.linear:
            sub_node = NLFragment(repn, node)
            sub_id = id(sub_node)
            sub_repn = AMPLRepn(0, None, None)
            sub_repn.nonlinear = repn.nonlinear
            sub_repn.nl = (visitor.template.var, (sub_id,))
            sub_repn.named_exprs = set(repn.named_exprs)
            repn.named_exprs.add(sub_id)
            repn.nonlinear = sub_repn.nl
            nl_info = list(expression_source)
            visitor.subexpression_cache[sub_id] = (sub_node, sub_repn, nl_info)
            visitor.subexpression_order.append(sub_id)
        else:
            nl_info = expression_source
    else:
        repn.nonlinear = None
        if repn.linear:
            if not repn.const and len(repn.linear) == 1 and (next(iter(repn.linear.values())) == 1):
                repn.nl = None
                expression_source[2] = True
        else:
            repn.nl = None
            expression_source[2] = True
    if mult != 1:
        repn.const *= mult
        if repn.linear:
            _lin = repn.linear
            for v in repn.linear:
                _lin[v] *= mult
        if repn.nonlinear:
            if mult == -1:
                prefix = visitor.template.negation
            else:
                prefix = visitor.template.multiplier % mult
            repn.nonlinear = (prefix + repn.nonlinear[0], repn.nonlinear[1])
    if expression_source[2]:
        if repn.linear:
            return (_MONOMIAL, next(iter(repn.linear)), 1)
        else:
            return (_CONSTANT, repn.const)
    visitor.subexpression_order.append(_id)
    return (_GENERAL, repn.duplicate())