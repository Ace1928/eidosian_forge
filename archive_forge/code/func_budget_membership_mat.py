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
@budget_membership_mat.setter
def budget_membership_mat(self, val):
    validate_array(arr=val, arr_name='budget_membership_mat', dim=2, valid_types=valid_num_types, valid_type_desc='a valid numeric type', required_shape=None)
    lhs_coeffs_arr = np.array(val)
    if hasattr(self, '_budget_membership_mat'):
        if lhs_coeffs_arr.shape[1] != self.dim:
            raise ValueError(f"BudgetSet attribute 'budget_membership_mat' must have {self.dim} columns to match set dimension (provided matrix with {lhs_coeffs_arr.shape[1]} columns)")
    if hasattr(self, '_budget_rhs_vec'):
        if lhs_coeffs_arr.shape[0] != self.budget_rhs_vec.size:
            raise ValueError(f"BudgetSet attribute 'budget_membership_mat' must have {self.budget_rhs_vec.size} rows to match shape of attribute 'budget_rhs_vec' (provided {lhs_coeffs_arr.shape[0]} rows)")
    uniq_entries = np.unique(lhs_coeffs_arr)
    non_bool_entries = uniq_entries[(uniq_entries != 0) & (uniq_entries != 1)]
    if non_bool_entries.size > 0:
        raise ValueError(f'Attempting to set attribute `budget_membership_mat` to value containing entries that are not 0-1 values (example: {non_bool_entries[0]}). Ensure all entries are of value 0 or 1')
    rows_with_zero_sums = np.nonzero(lhs_coeffs_arr.sum(axis=1) == 0)[0]
    if rows_with_zero_sums.size > 0:
        row_str = ', '.join((str(val) for val in rows_with_zero_sums))
        raise ValueError(f'Attempting to set attribute `budget_membership_mat` to value with all entries zero in rows at indexes: {row_str}. Ensure each row and column has at least one nonzero entry')
    cols_with_zero_sums = np.nonzero(lhs_coeffs_arr.sum(axis=0) == 0)[0]
    if cols_with_zero_sums.size > 0:
        col_str = ', '.join((str(val) for val in cols_with_zero_sums))
        raise ValueError(f'Attempting to set attribute `budget_membership_mat` to value with all entries zero in columns at indexes: {col_str}. Ensure each row and column has at least one nonzero entry')
    self._budget_membership_mat = lhs_coeffs_arr