from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.solvers.plugins.solvers.xpress_direct import XpressDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.numvalue import value, is_fixed
import pyomo.core.expr as EXPR
from pyomo.opt.base import SolverFactory
import collections
def get_xpress_control(self, *args):
    """
        Get xpress controls.

        Parameters
        ----------
        control(s): str, strs, list, None
            The xpress control to get. Options include any xpress control.
            Can also be list of xpress controls or None for every control
            Please see the Xpress documentation for options.

        See the Xpress documentation for xpress.problem.getControl for other
        uses of this function

        Returns
        -------
        control value or dictionary of control values
        """
    return self._solver_model.getControl(*args)