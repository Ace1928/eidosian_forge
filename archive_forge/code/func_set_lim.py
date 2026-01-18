import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def set_lim(min, max, name):
    """Set the domain bounds of the scale associated with the provided key.

    Parameters
    ----------
    name: hashable
        Any variable that can be used as a key for a dictionary

    Raises
    ------
    KeyError
        When no context figure is associated with the provided key.

    """
    scale = _context['scales'][_get_attribute_dimension(name)]
    scale.min = min
    scale.max = max
    return scale