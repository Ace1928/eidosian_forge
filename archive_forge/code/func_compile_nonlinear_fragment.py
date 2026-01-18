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
def compile_nonlinear_fragment(self, visitor):
    if not self.nonlinear:
        self.nonlinear = None
        return
    args = []
    nterms = len(self.nonlinear)
    nl_sum = ''.join(map(itemgetter(0), self.nonlinear))
    deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)
    if nterms > 2:
        self.nonlinear = (visitor.template.nary_sum % nterms + nl_sum, args)
    elif nterms == 2:
        self.nonlinear = (visitor.template.binary_sum + nl_sum, args)
    else:
        self.nonlinear = (nl_sum, args)