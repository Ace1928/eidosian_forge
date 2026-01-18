from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
from hvplot.plotting import hvPlot, hvPlotTabular
from holoviews import Store, Scatter
from holoviews.element.comparison import ComparisonTestCase
def test_define_custom_method(self):
    hvplot = hvPlotTabular(self.df, {'custom_scatter': {'width': 42, 'height': 42}})
    custom_scatter = hvplot.custom_scatter(y='y')
    scatter = hvplot.scatter(y='y')
    custom_opts = Store.lookup_options('bokeh', custom_scatter, 'plot')
    opts = Store.lookup_options('bokeh', scatter, 'plot')
    self.assertEqual(custom_opts.options.get('width'), 42)
    self.assertEqual(custom_opts.options.get('height'), 42)
    self.assertNotEqual(opts.options.get('width'), 42)
    self.assertNotEqual(opts.options.get('height'), 42)