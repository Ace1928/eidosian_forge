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
@budget_rhs_vec.setter
def budget_rhs_vec(self, val):
    validate_array(arr=val, arr_name='budget_rhs_vec', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
    rhs_vec_arr = np.array(val)
    if hasattr(self, '_budget_membership_mat'):
        if len(val) != self.budget_membership_mat.shape[0]:
            raise ValueError(f"Budget set attribute 'budget_rhs_vec' must have {self.budget_membership_mat.shape[0]} entries to match shape of attribute 'budget_membership_mat' (provided {rhs_vec_arr.size} entries)")
    for entry in rhs_vec_arr:
        if entry < 0:
            raise ValueError(f"Entry {entry} of attribute 'budget_rhs_vec' is negative. Ensure all entries are nonnegative")
    self._budget_rhs_vec = rhs_vec_arr