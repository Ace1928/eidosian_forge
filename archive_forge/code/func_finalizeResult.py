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
def finalizeResult(self, result):
    ans = node_result_to_amplrepn(result)
    ans.mult *= self.expression_scaling_factor
    if ans.nl is not None:
        if not ans.nl[1]:
            raise ValueError('Numeric expression resolved to a string constant')
        if not ans.linear:
            ans.named_exprs.update(ans.nl[1])
            ans.nonlinear = ans.nl
            ans.const = 0
        else:
            pass
        ans.nl = None
    if ans.nonlinear.__class__ is list:
        ans.compile_nonlinear_fragment(self)
    if not ans.linear:
        ans.linear = {}
    if ans.mult != 1:
        linear = ans.linear
        mult, ans.mult = (ans.mult, 1)
        ans.const *= mult
        if linear:
            for k in linear:
                linear[k] *= mult
        if ans.nonlinear:
            if mult == -1:
                prefix = self.template.negation
            else:
                prefix = self.template.multiplier % mult
            ans.nonlinear = (prefix + ans.nonlinear[0], ans.nonlinear[1])
    self.active_expression_source = None
    return ans