import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import HoloMap, NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Contours, Path, Polygons
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_batched_path_line_color_and_color(self):
    opts = {'NdOverlay': dict(legend_limit=0), 'Path': dict(line_color=Cycle(values=['red', 'blue']))}
    overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]]) for i in range(2)}).opts(opts)
    plot = bokeh_renderer.get_plot(overlay).subplots[()]
    line_color = ['red', 'blue']
    self.assertEqual(plot.handles['source'].data['line_color'], line_color)