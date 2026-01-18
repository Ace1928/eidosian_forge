import operator
import itertools
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.core import is_fixed, value
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.solvers.plugins.solvers.mosek_direct import MOSEKDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.opt.base import SolverFactory
from pyomo.core.kernel.conic import _ConicBase
from pyomo.core.kernel.block import block
def add_vars(self, var_seq):
    """
        Add multiple variables to the MOSEK task object in one method call.

        This will keep any existing model components intact.

        Parameters
        ----------
        var_seq: tuple/list of Var
        """
    self._add_vars(var_seq)