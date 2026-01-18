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
def evaluate_jacobian_external_variables(self):
    nlp = self._nlp
    x = self.input_vars
    y = self.external_vars
    g = self.external_cons
    jgx = nlp.extract_submatrix_jacobian(x, g)
    jgy = nlp.extract_submatrix_jacobian(y, g)
    jgy_csc = jgy.tocsc()
    dydx = -1 * sps.linalg.splu(jgy_csc).solve(jgx.toarray())
    return dydx