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
def boolean_values(self, variables: _IndexOrSeries) -> pd.Series:
    """Returns the values of the input variables.

        If `variables` is a `pd.Index`, then the output will be indexed by the
        variables. If `variables` is a `pd.Series` indexed by the underlying
        dimensions, then the output will be indexed by the same underlying
        dimensions.

        Args:
          variables (Union[pd.Index, pd.Series]): The set of variables from which to
            get the values.

        Returns:
          pd.Series: The values of all variables in the set.
        """
    solution = self._solution
    return _attribute_series(func=lambda literal: evaluate_boolean_expression(literal, solution), values=variables)