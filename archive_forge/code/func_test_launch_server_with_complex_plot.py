import asyncio
import time
import param
import pytest
from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.io.doc import curdoc, set_curdoc
from bokeh.models import ColumnDataSource
from panel import serve
from panel.io.state import state
from panel.widgets import DiscreteSlider, FloatSlider
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HLine, Path, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.plotting.bokeh.callbacks import Callback, RangeXYCallback, ResetCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import PlotReset, RangeXY, Stream
@pytest.mark.flaky(reruns=3)
def test_launch_server_with_complex_plot(self):
    dmap = DynamicMap(lambda x_range, y_range: Curve([]), streams=[RangeXY()])
    overlay = dmap * HLine(0)
    static = Polygons([]) * Path([]) * Curve([])
    layout = overlay + static
    self._launcher(layout, port=6003)