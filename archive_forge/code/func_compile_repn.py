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
def compile_repn(self, visitor, prefix='', args=None, named_exprs=None):
    template = visitor.template
    if self.mult != 1:
        if self.mult == -1:
            prefix += template.negation
        else:
            prefix += template.multiplier % self.mult
        self.mult = 1
    if self.named_exprs is not None:
        if named_exprs is None:
            named_exprs = set(self.named_exprs)
        else:
            named_exprs.update(self.named_exprs)
    if self.nl is not None:
        nl, nl_args = self.nl
        if prefix:
            nl = prefix + nl
        if args is not None:
            assert args is not nl_args
            args.extend(nl_args)
        else:
            args = list(nl_args)
        if nl_args:
            named_exprs.update(nl_args)
        return (nl, args, named_exprs)
    if args is None:
        args = []
    if self.linear:
        nterms = -len(args)
        _v_template = template.var
        _m_template = template.monomial
        nl_sum = ''.join((args.append(v) or (_v_template if c == 1 else _m_template % c) for v, c in self.linear.items() if c))
        nterms += len(args)
    else:
        nterms = 0
        nl_sum = ''
    if self.nonlinear:
        if self.nonlinear.__class__ is list:
            nterms += len(self.nonlinear)
            nl_sum += ''.join(map(itemgetter(0), self.nonlinear))
            deque(map(args.extend, map(itemgetter(1), self.nonlinear)), maxlen=0)
        else:
            nterms += 1
            nl_sum += self.nonlinear[0]
            args.extend(self.nonlinear[1])
    if self.const:
        nterms += 1
        nl_sum += template.const % self.const
    if nterms > 2:
        return (prefix + template.nary_sum % nterms + nl_sum, args, named_exprs)
    elif nterms == 2:
        return (prefix + template.binary_sum + nl_sum, args, named_exprs)
    elif nterms == 1:
        return (prefix + nl_sum, args, named_exprs)
    else:
        return (prefix + template.const % 0, args, named_exprs)