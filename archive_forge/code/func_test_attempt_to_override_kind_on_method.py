from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
from hvplot.plotting import hvPlot, hvPlotTabular
from holoviews import Store, Scatter
from holoviews.element.comparison import ComparisonTestCase
def test_attempt_to_override_kind_on_method(self):
    hvplot = hvPlotTabular(self.df, {'scatter': {'kind': 'line'}})
    self.assertIsInstance(hvplot.scatter(y='y'), Scatter)