from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
from hvplot.plotting import hvPlot, hvPlotTabular
from holoviews import Store, Scatter
from holoviews.element.comparison import ComparisonTestCase
def test_define_default_options(self):
    hvplot = hvPlotTabular(self.df, width=42, height=42)
    curve = hvplot(y='y')
    opts = Store.lookup_options('bokeh', curve, 'plot')
    self.assertEqual(opts.options.get('width'), 42)
    self.assertEqual(opts.options.get('height'), 42)