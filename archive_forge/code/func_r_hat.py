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
def r_hat(self):
    """Get rhat data for the variable."""
    _, y_vals, values, colors, model_names = self.labels_ticks_and_vals()
    for y, value, color, model_name in zip(y_vals, values, colors, model_names):
        if value.ndim != 2 or value.shape[0] < 2:
            yield (y, None, color, model_name)
        else:
            yield (y, _rhat(value), color, model_name)