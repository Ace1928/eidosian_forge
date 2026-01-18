from __future__ import annotations
import warnings
from typing import List, Tuple, TypeVar
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
def as_quad_approx(self, m: int, k: int) -> RelEntrConeQuad:
    return RelEntrConeQuad(self.y, self.z, -self.x, m, k)