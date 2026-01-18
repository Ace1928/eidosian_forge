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
def test_launch_server_with_stream(self):
    el = Curve([])
    stream = RangeXY(source=el)
    obj, _ = bokeh_renderer._validate(el, None)
    server, _ = self._launcher(obj, port=6002)
    [(plot, _)] = obj._plots.values()
    cb = plot.callbacks[0]
    self.assertIsInstance(cb, RangeXYCallback)
    self.assertEqual(cb.streams, [stream])
    assert 'rangesupdate' in plot.state._event_callbacks