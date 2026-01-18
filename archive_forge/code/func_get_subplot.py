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
def get_subplot(self, row, col, secondary_y=False):
    """
        Return an object representing the subplot at the specified row
        and column.  May only be used on Figures created using
        plotly.tools.make_subplots

        Parameters
        ----------
        row: int
            1-based index of subplot row
        col: int
            1-based index of subplot column
        secondary_y: bool
            If True, select the subplot that consists of the x-axis and the
            secondary y-axis at the specified row/col. Only valid if the
            subplot at row/col is an 2D cartesian subplot that was created
            with a secondary y-axis.  See the docstring for the specs argument
            to make_subplots for more info on creating a subplot with a
            secondary y-axis.
        Returns
        -------
        subplot
            * None: if subplot is empty
            * plotly.graph_objs.layout.Scene: if subplot type is 'scene'
            * plotly.graph_objs.layout.Polar: if subplot type is 'polar'
            * plotly.graph_objs.layout.Ternary: if subplot type is 'ternary'
            * plotly.graph_objs.layout.Mapbox: if subplot type is 'ternary'
            * SubplotDomain namedtuple with `x` and `y` fields:
              if subplot type is 'domain'.
                - x: length 2 list of the subplot start and stop width
                - y: length 2 list of the subplot start and stop height
            * SubplotXY namedtuple with `xaxis` and `yaxis` fields:
              if subplot type is 'xy'.
                - xaxis: plotly.graph_objs.layout.XAxis instance for subplot
                - yaxis: plotly.graph_objs.layout.YAxis instance for subplot
        """
    from plotly._subplots import _get_grid_subplot
    return _get_grid_subplot(self, row, col, secondary_y)