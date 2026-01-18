from collections.abc import Iterable
import logging
import math
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.log import LogStream
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import Var, _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.core.staleflag import StaleFlagManager
import sys
def get_updated_expression(self):
    for ndx, coef in enumerate(self.linear_coefs):
        coef.var.obj = value(coef.expr)
    self.gurobi_model.ObjCon = value(self.constant.expr)
    gurobi_expr = None
    for ndx, coef in enumerate(self.quadratic_coefs):
        if value(coef.expr) != self.last_quadratic_coef_values[ndx]:
            if gurobi_expr is None:
                self.gurobi_model.update()
                gurobi_expr = self.gurobi_model.getObjective()
            current_coef_value = value(coef.expr)
            incremental_coef_value = current_coef_value - self.last_quadratic_coef_values[ndx]
            gurobi_expr += incremental_coef_value * coef.var1 * coef.var2
            self.last_quadratic_coef_values[ndx] = current_coef_value
    return gurobi_expr