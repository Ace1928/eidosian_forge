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
def plotly_restyle(self, restyle_data, trace_indexes=None, **kwargs):
    """
        Perform a Plotly restyle operation on the figure's traces

        Parameters
        ----------
        restyle_data : dict
            Dict of trace style updates.

            Keys are strings that specify the properties to be updated.
            Nested properties are expressed by joining successive keys on
            '.' characters (e.g. 'marker.color').

            Values may be scalars or lists. When values are scalars,
            that scalar value is applied to all traces specified by the
            `trace_indexes` parameter.  When values are lists,
            the restyle operation will cycle through the elements
            of the list as it cycles through the traces specified by the
            `trace_indexes` parameter.

            Caution: To use plotly_restyle to update a list property (e.g.
            the `x` property of the scatter trace), the property value
            should be a scalar list containing the list to update with. For
            example, the following command would be used to update the 'x'
            property of the first trace to the list [1, 2, 3]

            >>> import plotly.graph_objects as go
            >>> fig = go.Figure(go.Scatter(x=[2, 4, 6]))
            >>> fig.plotly_restyle({'x': [[1, 2, 3]]}, 0)

        trace_indexes : int or list of int
            Trace index, or list of trace indexes, that the restyle operation
            applies to. Defaults to all trace indexes.

        Returns
        -------
        None
        """
    trace_indexes = self._normalize_trace_indexes(trace_indexes)
    source_view_id = kwargs.get('source_view_id', None)
    restyle_changes = self._perform_plotly_restyle(restyle_data, trace_indexes)
    if restyle_changes:
        msg_kwargs = {'source_view_id': source_view_id} if source_view_id is not None else {}
        self._send_restyle_msg(restyle_changes, trace_indexes=trace_indexes, **msg_kwargs)
        self._dispatch_trace_change_callbacks(restyle_changes, trace_indexes)