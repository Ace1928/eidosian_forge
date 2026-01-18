import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_axis_labels(self):
    overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True) for i in range(2)])
    plot = bokeh_renderer.get_plot(overlay)
    assert plot.state.xaxis.axis_label == 'x'
    assert plot.state.yaxis.axis_label == 'y'