import itertools
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.timing import HierarchicalTimer
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
import numpy as np
import scipy.sparse as sps
def set_equality_constraint_scaling_factors(self, scaling_factors):
    """
        Set scaling factors for the equality constraints that are exposed
        to a solver. These are the "residual equations" in this class.
        """
    self.residual_scaling_factors = np.array(scaling_factors)