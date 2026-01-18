from pyomo.common.dependencies import attempt_import
import itertools
import logging
from operator import attrgetter
from pyomo.common import DeveloperError
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.collections import ComponentMap
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.interval_var import (
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.core.base import (
from pyomo.core.base.boolean_var import (
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.param import IndexedParam, ScalarParam, _ParamData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
import pyomo.core.expr as EXPR
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.core.base import Set, RangeSet
from pyomo.core.base.set import SetProduct
from pyomo.opt import WriterFactory, SolverFactory, TerminationCondition, SolverResults
from pyomo.network import Port
def collect_valid_components(model, active=True, sort=None, valid=set(), targets=set()):
    assert active in (True, None)
    unrecognized = {}
    components = {k: [] for k in targets}
    for obj in model.component_data_objects(active=True, descend_into=True, sort=sort):
        ctype = obj.ctype
        if ctype in components:
            components[ctype].append(obj)
        elif ctype not in valid:
            if ctype not in unrecognized:
                unrecognized[ctype] = [obj]
            else:
                unrecognized[ctype].append(obj)
    return (components, unrecognized)