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
def handle_external_function_node(visitor, node, *args):
    func = node._fcn._function
    if all((arg[0] is _CONSTANT or (arg[0] is _GENERAL and arg[1].nl and (not arg[1].nl[1])) for arg in args)):
        arg_list = [arg[1] if arg[0] is _CONSTANT else arg[1].const for arg in args]
        return (_CONSTANT, apply_node_operation(node, arg_list))
    if func in visitor.external_functions:
        if node._fcn._library != visitor.external_functions[func][1]._library:
            raise RuntimeError('The same external function name (%s) is associated with two different libraries (%s through %s, and %s through %s).  The ASL solver will fail to link correctly.' % (func, visitor.external_byFcn[func]._library, visitor.external_byFcn[func]._library.name, node._fcn._library, node._fcn.name))
    else:
        visitor.external_functions[func] = (len(visitor.external_functions), node._fcn)
    comment = f'\t#{node.local_name}' if visitor.symbolic_solver_labels else ''
    nonlin = node_result_to_amplrepn(args[0]).compile_repn(visitor, visitor.template.external_fcn % (visitor.external_functions[func][0], len(args), comment))
    for arg in args[1:]:
        nonlin = node_result_to_amplrepn(arg).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))