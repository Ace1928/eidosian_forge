import datetime as dt
import numpy as np
import pandas as pd
import pytest
from holoviews.core.options import AbbreviatedException
from holoviews.core.overlay import NdOverlay
from holoviews.element import Curve
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
def test_curve_linewidth_op(self):
    curve = Curve([(0, 0, 0.1), (0, 1, 0.3), (0, 2, 1)], vdims=['y', 'linewidth']).opts(linewidth='linewidth')
    msg = 'ValueError: Mapping a dimension to the "linewidth" style'
    with pytest.raises(AbbreviatedException, match=msg):
        mpl_renderer.get_plot(curve)