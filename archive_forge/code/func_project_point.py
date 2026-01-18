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
def project_point(cb, msg, attributes=('x', 'y')):
    """
    Projects a single point supplied by a callback
    """
    if skip(cb, msg, attributes):
        return msg
    plot = get_cb_plot(cb)
    x, y = (msg.get('x', 0), msg.get('y', 0))
    crs = plot.current_frame.crs
    coordinates = crs.transform_points(plot.projection, np.array([x]), np.array([y]))
    msg['x'], msg['y'] = coordinates[0, :2]
    return {k: v for k, v in msg.items() if k in attributes}