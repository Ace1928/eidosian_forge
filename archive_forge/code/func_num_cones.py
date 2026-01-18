from typing import List, Tuple
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
def num_cones(self):
    return self.z.size