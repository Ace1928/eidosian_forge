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
def add_no_overlap_2d(self, x_intervals: Iterable[IntervalVar], y_intervals: Iterable[IntervalVar]) -> Constraint:
    """Adds NoOverlap2D(x_intervals, y_intervals).

        A NoOverlap2D constraint ensures that all present rectangles do not overlap
        on a plane. Each rectangle is aligned with the X and Y axis, and is defined
        by two intervals which represent its projection onto the X and Y axis.

        Furthermore, one box is optional if at least one of the x or y interval is
        optional.

        Args:
          x_intervals: The X coordinates of the rectangles.
          y_intervals: The Y coordinates of the rectangles.

        Returns:
          An instance of the `Constraint` class.
        """
    ct = Constraint(self)
    model_ct = self.__model.constraints[ct.index]
    model_ct.no_overlap_2d.x_intervals.extend([self.get_interval_index(x) for x in x_intervals])
    model_ct.no_overlap_2d.y_intervals.extend([self.get_interval_index(x) for x in y_intervals])
    return ct