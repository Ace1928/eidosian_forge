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
@shape_matrix.setter
def shape_matrix(self, val):
    validate_array(arr=val, arr_name='shape_matrix', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
    shape_mat_arr = np.array(val)
    if hasattr(self, '_center'):
        if not all((size == self.dim for size in shape_mat_arr.shape)):
            raise ValueError(f"EllipsoidalSet attribute 'shape_matrix' must be a square matrix of size {self.dim} to match set dimension (provided matrix with shape {shape_mat_arr.shape})")
    self._verify_positive_definite(shape_mat_arr)
    self._shape_matrix = shape_mat_arr