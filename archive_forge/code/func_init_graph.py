import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
def init_graph(self, datum, options, index=0, **kwargs):
    """
        Initialize the plotly components that will represent the element

        Parameters
        ----------
        datum: dict
            An element of the data list returned by the get_data method
        options: dict
            Graph options that were returned by the graph_options method
        index: int
            Index of datum in the original list returned by the get_data method

        Returns
        -------
        dict
            Dictionary of the plotly components that represent the element.
            Keys may include:
             - 'traces': List of trace dicts
             - 'annotations': List of annotations dicts
             - 'images': List of image dicts
             - 'shapes': List of shape dicts
        """
    trace = dict(options)
    for k, v in datum.items():
        if k in trace and isinstance(trace[k], dict):
            trace[k].update(v)
        else:
            trace[k] = v
    if self._style_key and self._per_trace:
        vectorized = {k: v for k, v in options[self._style_key].items() if isinstance(v, np.ndarray)}
        trace[self._style_key] = dict(trace[self._style_key])
        for s, val in vectorized.items():
            trace[self._style_key][s] = val[index]
    return {'traces': [trace]}