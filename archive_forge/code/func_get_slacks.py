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
def get_slacks(self, cons_to_load=None):
    if self._needs_updated:
        self._update_gurobi_model()
    if self._solver_model.SolCount == 0:
        raise RuntimeError('Solver does not currently have valid slacks. Please check the termination condition.')
    con_map = self._pyomo_con_to_solver_con_map
    reverse_con_map = self._solver_con_to_pyomo_con_map
    slack = dict()
    gurobi_range_con_vars = OrderedSet(self._solver_model.getVars()) - OrderedSet(self._pyomo_var_to_solver_var_map.values())
    if cons_to_load is None:
        linear_cons_to_load = self._solver_model.getConstrs()
        quadratic_cons_to_load = self._solver_model.getQConstrs()
    else:
        gurobi_cons_to_load = OrderedSet([con_map[pyomo_con] for pyomo_con in cons_to_load])
        linear_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getConstrs())))
        quadratic_cons_to_load = list(gurobi_cons_to_load.intersection(OrderedSet(self._solver_model.getQConstrs())))
    linear_vals = self._solver_model.getAttr('Slack', linear_cons_to_load)
    quadratic_vals = self._solver_model.getAttr('QCSlack', quadratic_cons_to_load)
    for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
        pyomo_con = reverse_con_map[id(gurobi_con)]
        if pyomo_con in self._range_constraints:
            lin_expr = self._solver_model.getRow(gurobi_con)
            for i in reversed(range(lin_expr.size())):
                v = lin_expr.getVar(i)
                if v in gurobi_range_con_vars:
                    Us_ = v.X
                    Ls_ = v.UB - v.X
                    if Us_ > Ls_:
                        slack[pyomo_con] = Us_
                    else:
                        slack[pyomo_con] = -Ls_
                    break
        else:
            slack[pyomo_con] = val
    for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
        pyomo_con = reverse_con_map[id(gurobi_con)]
        slack[pyomo_con] = val
    return slack