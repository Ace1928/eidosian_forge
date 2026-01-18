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
def calculate_external_constraint_multipliers(self, resid_multipliers):
    """
        Calculates the multipliers of the external constraints from the
        multipliers of the residual constraints (which are provided by
        the "outer" solver).

        """
    nlp = self._nlp
    y = self.external_vars
    f = self.residual_cons
    g = self.external_cons
    jfy = nlp.extract_submatrix_jacobian(y, f)
    jgy = nlp.extract_submatrix_jacobian(y, g)
    jgy_t = jgy.transpose()
    jfy_t = jfy.transpose()
    dfdg = -sps.linalg.splu(jgy_t.tocsc()).solve(jfy_t.toarray())
    resid_multipliers = np.array(resid_multipliers)
    external_multipliers = dfdg.dot(resid_multipliers)
    return external_multipliers