import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
def assert_output_options_group_empty(self, obj):
    mpl_output_lookup = Store.lookup_options('matplotlib', obj, 'output').options
    self.assertEqual(mpl_output_lookup, {})
    bokeh_output_lookup = Store.lookup_options('bokeh', obj, 'output').options
    self.assertEqual(bokeh_output_lookup, {})