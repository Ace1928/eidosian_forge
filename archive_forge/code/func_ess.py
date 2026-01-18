from collections import OrderedDict, defaultdict
from itertools import cycle, tee
import bokeh.plotting as bkp
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import Band, ColumnDataSource, DataRange1d
from bokeh.models.annotations import Title, Legend
from bokeh.models.tickers import FixedTicker
from ....sel_utils import xarray_var_iter
from ....rcparams import rcParams
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ....stats.diagnostics import _ess, _rhat
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults
def ess(self):
    """Get effective n data for the variable."""
    _, y_vals, values, colors, model_names = self.labels_ticks_and_vals()
    for y, value, color, model_name in zip(y_vals, values, colors, model_names):
        yield (y, _ess(value), color, model_name)