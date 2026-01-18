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
@positive_deviation.setter
def positive_deviation(self, val):
    validate_array(arr=val, arr_name='positive_deviation', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type')
    for dev_val in val:
        if dev_val < 0:
            raise ValueError(f"Entry {dev_val} of attribute 'positive_deviation' is negative value")
    val_arr = np.array(val)
    if hasattr(self, '_origin'):
        if val_arr.size != self.dim:
            raise ValueError(f"Attempting to set attribute 'positive_deviation' of cardinality set of dimension {self.dim} to value of dimension {val_arr.size}")
    self._positive_deviation = val_arr