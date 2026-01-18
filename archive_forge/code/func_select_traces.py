import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def select_traces(self, selector=None, row=None, col=None, secondary_y=None):
    """
        Select traces from a particular subplot cell and/or traces
        that satisfy custom selection criteria.

        Parameters
        ----------
        selector: dict, function, int, str or None (default None)
            Dict to use as selection criteria.
            Traces will be selected if they contain properties corresponding
            to all of the dictionary's keys, with values that exactly match
            the supplied values. If None (the default), all traces are
            selected. If a function, it must be a function accepting a single
            argument and returning a boolean. The function will be called on
            each trace and those for which the function returned True
            will be in the selection. If an int N, the Nth trace matching row
            and col will be selected (N can be negative). If a string S, the selector
            is equivalent to dict(type=S).
        row, col: int or None (default None)
            Subplot row and column index of traces to select.
            To select traces by row and column, the Figure must have been
            created using plotly.subplots.make_subplots.  If None
            (the default), all traces are selected.
        secondary_y: boolean or None (default None)
            * If True, only select traces associated with the secondary
              y-axis of the subplot.
            * If False, only select traces associated with the primary
              y-axis of the subplot.
            * If None (the default), do not filter traces based on secondary
              y-axis.

            To select traces by secondary y-axis, the Figure must have been
            created using plotly.subplots.make_subplots. See the docstring
            for the specs argument to make_subplots for more info on
            creating subplots with secondary y-axes.
        Returns
        -------
        generator
            Generator that iterates through all of the traces that satisfy
            all of the specified selection criteria
        """
    if not selector and (not isinstance(selector, int)):
        selector = {}
    if row is not None or col is not None or secondary_y is not None:
        grid_ref = self._validate_get_grid_ref()
        filter_by_subplot = True
        if row is None and col is not None:
            grid_subplot_ref_tuples = [ref_row[col - 1] for ref_row in grid_ref]
        elif col is None and row is not None:
            grid_subplot_ref_tuples = grid_ref[row - 1]
        elif col is not None and row is not None:
            grid_subplot_ref_tuples = [grid_ref[row - 1][col - 1]]
        else:
            grid_subplot_ref_tuples = [refs for refs_row in grid_ref for refs in refs_row]
        grid_subplot_refs = []
        for refs in grid_subplot_ref_tuples:
            if not refs:
                continue
            if secondary_y is not True:
                grid_subplot_refs.append(refs[0])
            if secondary_y is not False and len(refs) > 1:
                grid_subplot_refs.append(refs[1])
    else:
        filter_by_subplot = False
        grid_subplot_refs = None
    return self._perform_select_traces(filter_by_subplot, grid_subplot_refs, selector)