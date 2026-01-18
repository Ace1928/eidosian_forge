from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.kernel.suffix import import_suffix_generator
from pyomo.core.expr.numvalue import native_numeric_types, value
from pyomo.core.expr.visitor import evaluate_expression
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
import time
import logging
def remove_sos_constraint(self, con):
    """Remove a single SOS constraint from the solver's model.

        This will keep any other model components intact.

        Parameters
        ----------
        con: SOSConstraint

        """
    solver_con = self._pyomo_con_to_solver_con_map[con]
    self._remove_sos_constraint(solver_con)
    self._symbol_map.removeSymbol(con)
    for var in self._vars_referenced_by_con[con]:
        self._referenced_variables[var] -= 1
    del self._vars_referenced_by_con[con]
    del self._pyomo_con_to_solver_con_map[con]
    del self._solver_con_to_pyomo_con_map[solver_con]