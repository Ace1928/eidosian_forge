import abc
from typing import List
from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.contrib.solver.util import collect_vars_and_named_exprs, get_objective
def add_sos_constraints(self, cons: List[_SOSConstraintData]):
    for con in cons:
        if con in self._vars_referenced_by_con:
            raise ValueError('constraint {name} has already been added'.format(name=con.name))
        self._active_constraints[con] = tuple()
        variables = con.get_variables()
        self._check_for_new_vars(variables)
        self._named_expressions[con] = []
        self._vars_referenced_by_con[con] = variables
        for v in variables:
            self._referenced_variables[id(v)][1][con] = None
    self._add_sos_constraints(cons)