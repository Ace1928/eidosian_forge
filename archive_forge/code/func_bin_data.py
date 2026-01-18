import os
import json
from warnings import warn
import ipywidgets as widgets
from ipywidgets import (Widget, DOMWidget, CallbackDispatcher,
from traitlets import (Int, Unicode, List, Enum, Dict, Bool, Float,
from traittypes import Array
from numpy import histogram
import numpy as np
from .scales import Scale, OrdinalScale, LinearScale
from .traits import (Date, array_serialization,
from ._version import __frontend_version__
from .colorschemes import CATEGORY10
def bin_data(self, *args):
    """
        Performs the binning of `sample` data, and draws the corresponding bars
        """
    _min = self.sample.min() if self.min is None else self.min
    _max = self.sample.max() if self.max is None else self.max
    _range = (min(_min, _max), max(_min, _max))
    counts, bin_edges = histogram(self.sample, bins=self.bins, range=_range, density=self.density)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    with self.hold_sync():
        self.x, self.y = (midpoints, counts)