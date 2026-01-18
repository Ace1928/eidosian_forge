import datetime as dt
import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_area_legend(self):
    python = np.array([2, 3, 7, 5, 26, 221, 44, 233, 254, 265, 266, 267, 120, 111])
    pypy = np.array([12, 33, 47, 15, 126, 121, 144, 233, 254, 225, 226, 267, 110, 130])
    jython = np.array([22, 43, 10, 25, 26, 101, 114, 203, 194, 215, 201, 227, 139, 160])
    dims = dict(kdims='time', vdims='memory')
    python = Area(python, label='python', **dims)
    pypy = Area(pypy, label='pypy', **dims)
    jython = Area(jython, label='jython', **dims)
    overlay = Area.stack(python * pypy * jython)
    labels = [n[1] for n in overlay.data]
    self.assertEqual(labels, ['Python', 'Pypy', 'Jython'])