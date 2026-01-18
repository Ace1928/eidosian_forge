from io import StringIO
from typing import Sequence, Dict, Optional, Mapping, MutableMapping
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver import results
from pyomo.contrib.solver import solution
import pyomo.environ as pyo
from pyomo.core.base.var import Var
def get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
    if self._reduced_costs is None:
        raise RuntimeError('Solution loader does not currently have valid reduced costs. Please check the termination condition and ensure the solver returns reduced costs for the given problem type.')
    if vars_to_load is None:
        rc = ComponentMap(self._reduced_costs.values())
    else:
        rc = ComponentMap()
        for v in vars_to_load:
            rc[v] = self._reduced_costs[id(v)][1]
    return rc