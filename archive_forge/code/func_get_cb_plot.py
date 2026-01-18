import numpy as np
from pathlib import Path
from bokeh.models import CustomJS, CustomAction, PolyEditTool
from holoviews.core.ndmapping import UniformNdMapping
from holoviews.plotting.bokeh.callbacks import (
from holoviews.streams import (
from ...element.geo import _Element, Shape
from ...util import project_extents
from ...models import PolyVertexDrawTool, PolyVertexEditTool
from ...operation import project
from ...streams import PolyVertexEdit, PolyVertexDraw
from .plot import GeoOverlayPlot
def get_cb_plot(cb, plot=None):
    """
    Finds the subplot with the corresponding stream.
    """
    plot = plot or cb.plot
    if isinstance(plot, GeoOverlayPlot):
        plots = [get_cb_plot(cb, p) for p in plot.subplots.values()]
        plots = [p for p in plots if any((s in cb.streams and getattr(s, '_triggering', False) for s in p.streams))]
        if plots:
            plot = plots[0]
    return plot