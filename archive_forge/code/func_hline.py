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
def hline(level, **kwargs):
    """Draws a horizontal line at the given level.

    Parameters
    ----------
    level: float
        The level at which to draw the horizontal line.
    preserve_domain: boolean (default: False)
        If true, the line does not affect the domain of the 'y' scale.
    """
    kwargs.setdefault('colors', ['dodgerblue'])
    kwargs.setdefault('stroke_width', 1)
    scales = kwargs.pop('scales', {})
    fig = kwargs.get('figure', current_figure())
    scales['x'] = fig.scale_x
    level = array(level)
    if len(level.shape) == 0:
        x = [0, 1]
        y = [level, level]
    else:
        x = [0, 1]
        y = column_stack([level, level])
    return plot(x, y, scales=scales, preserve_domain={'x': True, 'y': kwargs.get('preserve_domain', False)}, axes=False, update_context=False, **kwargs)