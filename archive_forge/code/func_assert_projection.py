import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def assert_projection(self, plot, proj):
    opts = hv.Store.lookup_options('bokeh', plot, 'plot')
    assert opts.kwargs['projection'].proj4_params['proj'] == proj