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
def set_instance(self, model, **kwds):
    """
        This method is used to translate the Pyomo model provided to an instance of the solver's Python model. This
        discards any existing model and starts from scratch.

        Parameters
        ----------
        model: ConcreteModel
            The pyomo model to be used with the solver.

        Keyword Arguments
        -----------------
        symbolic_solver_labels: bool
            If True, the solver's components (e.g., variables, constraints) will be given names that correspond to
            the Pyomo component names.
        skip_trivial_constraints: bool
            If True, then any constraints with a constant body will not be added to the solver model.
            Be careful with this. If a trivial constraint is skipped then that constraint cannot be removed from
            a persistent solver (an error will be raised if a user tries to remove a non-existent constraint).
        output_fixed_variable_bounds: bool
            If False then an error will be raised if a fixed variable is used in one of the solver constraints.
            This is useful for catching bugs. Ordinarily a fixed variable should appear as a constant value in the
            solver constraints. If True, then the error will not be raised.
        """
    return self._set_instance(model, kwds)