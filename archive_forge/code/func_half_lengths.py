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
@half_lengths.setter
def half_lengths(self, val):
    validate_array(arr=val, arr_name='half_lengths', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
    val_arr = np.array(val)
    if hasattr(self, '_center'):
        if val_arr.size != self.dim:
            raise ValueError(f"Attempting to set attribute 'half_lengths' of AxisAlignedEllipsoidalSet of dimension {self.dim} to value of dimension {val_arr.size}")
    for half_len in val_arr:
        if half_len < 0:
            raise ValueError(f"Entry {half_len} of 'half_lengths' is negative. All half-lengths must be nonnegative")
    self._half_lengths = val_arr