from typing import List, Tuple
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
def save_dual_value(self, value) -> None:
    dW = value[:, :-1]
    dz = value[:, -1]
    if self.axis == 0:
        dW = dW.T
        dz = dz.T
    if dW.shape[1] == 1:
        dW = np.squeeze(dW)
    self.dual_variables[0].save_value(dW)
    self.dual_variables[1].save_value(dz)