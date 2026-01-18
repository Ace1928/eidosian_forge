from pyomo.contrib.appsi.base import PersistentBase
from pyomo.common.config import (
from .cmodel import cmodel, cmodel_available
from typing import List, Optional
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData, minimize, maximize
from pyomo.core.base.block import _BlockData
from pyomo.core.base import SymbolMap, TextLabeler
from pyomo.common.errors import InfeasibleConstraintException
def perform_fbbt_with_seed(self, model: _BlockData, seed_var: _GeneralVarData):
    if model is not self._model:
        self.set_instance(model)
    else:
        self.update()
    try:
        n_iter = self._cmodel.perform_fbbt_with_seed(self._var_map[id(seed_var)], self.config.feasibility_tol, self.config.integer_tol, self.config.improvement_tol, self.config.max_iter, self.config.deactivate_satisfied_constraints)
    finally:
        self._update_pyomo_var_bounds()
        self._deactivate_satisfied_cons()
    return n_iter