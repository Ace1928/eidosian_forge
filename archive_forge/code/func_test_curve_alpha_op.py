import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models import FactorRange, FixedTicker
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.options import AbbreviatedException, Cycle, Palette
from holoviews.element import Curve
from holoviews.plotting.bokeh.callbacks import Callback, PointerXCallback
from holoviews.plotting.util import rgb2hex
from holoviews.streams import PointerX
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_curve_alpha_op(self):
    curve = Curve([(0, 0, 0.1), (0, 1, 0.3), (0, 2, 1)], vdims=['y', 'alpha']).opts(alpha='alpha')
    msg = 'ValueError: Mapping a dimension to the "alpha" style'
    with pytest.raises(AbbreviatedException, match=msg):
        bokeh_renderer.get_plot(curve)