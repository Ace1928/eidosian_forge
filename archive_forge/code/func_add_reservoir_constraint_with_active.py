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
def add_reservoir_constraint_with_active(self, times: Iterable[LinearExprT], level_changes: Iterable[LinearExprT], actives: Iterable[LiteralT], min_level: int, max_level: int) -> Constraint:
    """Adds Reservoir(times, level_changes, actives, min_level, max_level).

        Maintains a reservoir level within bounds. The water level starts at 0, and
        at any time, it must be between min_level and max_level.

        If the variable `times[i]` is assigned a value t, and `actives[i]` is
        `True`, then the current level changes by `level_changes[i]`, which is
        constant,
        at time t.

         Note that min level must be <= 0, and the max level must be >= 0. Please
         use fixed level_changes to simulate initial state.

         Therefore, at any time:
             sum(level_changes[i] * actives[i] if times[i] <= t) in [min_level,
             max_level]


        The array of boolean variables 'actives', if defined, indicates which
        actions are actually performed.

        Args:
          times: A list of 1-var affine expressions (a * x + b) which specify the
            time of the filling or emptying the reservoir.
          level_changes: A list of integer values that specifies the amount of the
            emptying or filling. Currently, variable demands are not supported.
          actives: a list of boolean variables. They indicates if the
            emptying/refilling events actually take place.
          min_level: At any time, the level of the reservoir must be greater or
            equal than the min level.
          max_level: At any time, the level of the reservoir must be less or equal
            than the max level.

        Returns:
          An instance of the `Constraint` class.

        Raises:
          ValueError: if max_level < min_level.

          ValueError: if max_level < 0.

          ValueError: if min_level > 0
        """
    if max_level < min_level:
        raise ValueError('Reservoir constraint must have a max_level >= min_level')
    if max_level < 0:
        raise ValueError('Reservoir constraint must have a max_level >= 0')
    if min_level > 0:
        raise ValueError('Reservoir constraint must have a min_level <= 0')
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.reservoir.time_exprs.extend([self.parse_linear_expression(x) for x in times])
    model_ct.reservoir.level_changes.extend([self.parse_linear_expression(x) for x in level_changes])
    model_ct.reservoir.active_literals.extend([self.get_or_make_boolean_index(x) for x in actives])
    model_ct.reservoir.min_level = min_level
    model_ct.reservoir.max_level = max_level
    return ct