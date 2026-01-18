from pyomo.core.base.PyomoModel import Model
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.opt.base.solvers import OptSolver
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
import pyomo.common
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt.base.solvers
from pyomo.opt.base.formats import ResultsFormat
from pyomo.core.staleflag import StaleFlagManager
def load_vars(self, vars_to_load=None):
    """
        Load the values from the solver's variables into the corresponding pyomo variables.

        Parameters
        ----------
        vars_to_load: list of Var
        """
    self._load_vars(vars_to_load)
    StaleFlagManager.mark_all_as_stale(delayed=True)