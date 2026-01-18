import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def new_optional_interval_var_series(self, name: str, index: pd.Index, starts: Union[LinearExprT, pd.Series], sizes: Union[LinearExprT, pd.Series], ends: Union[LinearExprT, pd.Series], are_present: Union[LiteralT, pd.Series]) -> pd.Series:
    """Creates a series of interval variables with the given name.

        Args:
          name (str): Required. The name of the variable set.
          index (pd.Index): Required. The index to use for the variable set.
          starts (Union[LinearExprT, pd.Series]): The start of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          sizes (Union[LinearExprT, pd.Series]): The size of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          ends (Union[LinearExprT, pd.Series]): The ends of each interval in the
            set. If a `pd.Series` is passed in, it will be based on the
            corresponding values of the pd.Series.
          are_present (Union[LiteralT, pd.Series]): The performed literal of each
            interval in the set. If a `pd.Series` is passed in, it will be based on
            the corresponding values of the pd.Series.

        Returns:
          pd.Series: The interval variable set indexed by its corresponding
          dimensions.

        Raises:
          TypeError: if the `index` is invalid (e.g. a `DataFrame`).
          ValueError: if the `name` is not a valid identifier or already exists.
          ValueError: if the all the indexes do not match.
        """
    if not isinstance(index, pd.Index):
        raise TypeError('Non-index object is used as index')
    if not name.isidentifier():
        raise ValueError('name={} is not a valid identifier'.format(name))
    starts = _convert_to_linear_expr_series_and_validate_index(starts, index)
    sizes = _convert_to_linear_expr_series_and_validate_index(sizes, index)
    ends = _convert_to_linear_expr_series_and_validate_index(ends, index)
    are_present = _convert_to_literal_series_and_validate_index(are_present, index)
    interval_array = []
    for i in index:
        interval_array.append(self.new_optional_interval_var(start=starts[i], size=sizes[i], end=ends[i], is_present=are_present[i], name=f'{name}[{i}]'))
    return pd.Series(index=index, data=interval_array)