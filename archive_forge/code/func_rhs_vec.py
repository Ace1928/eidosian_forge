import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
@property
def rhs_vec(self):
    """
        (L + N,) numpy.ndarray : Right-hand side vector for polyhedral
        constraints defining the budget set. This also includes entries
        for nonnegativity constraints on the uncertain parameters.

        This attribute cannot be set, and is automatically determined
        given other attributes.
        """
    return np.append(self.budget_rhs_vec + self.budget_membership_mat @ self.origin, -self.origin)