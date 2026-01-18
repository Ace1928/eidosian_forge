import numpy as np
import pytest
from holoviews.core import Overlay
from holoviews.element import Curve
from holoviews.element.annotation import VSpan
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bool_scale(self):
    test_data = [(0.5, (-0.25, 0.25), (0.75, 1.25), (-0.25, 1.25)), (1, (-0.5, 0.5), (0.5, 1.5), (-0.5, 1.5)), (2, (-1, 1), (0, 2), (-1, 2)), (5, (-2.5, 2.5), (-1.5, 3.5), (-2.5, 3.5))]
    for scale, ytarget1, ytarget2, ytarget in test_data:
        overlay = Overlay([Curve(range(10), label=f'Data {i}').opts(subcoordinate_y=True, subcoordinate_scale=scale) for i in range(2)])
        plot = bokeh_renderer.get_plot(overlay)
        sp1 = plot.subplots['Curve', 'Data_0']
        assert sp1.handles['glyph_renderer'].coordinates.y_target.start == ytarget1[0]
        assert sp1.handles['glyph_renderer'].coordinates.y_target.end == ytarget1[1]
        sp2 = plot.subplots['Curve', 'Data_1']
        assert sp2.handles['glyph_renderer'].coordinates.y_target.start == ytarget2[0]
        assert sp2.handles['glyph_renderer'].coordinates.y_target.end == ytarget2[1]
        assert plot.handles['y_range'].start == ytarget[0]
        assert plot.handles['y_range'].end == ytarget[1]