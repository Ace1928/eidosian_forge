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
def plot_rhat(self, ax, markersize, plotted):
    """Draw r-hat for each plotter."""
    for plotter in self.plotters.values():
        for y, r_hat, color, model_name in plotter.r_hat():
            if r_hat is not None:
                plotted[model_name].append(ax.circle(x=r_hat, y=y, fill_color=color, size=markersize, line_color='black'))
    ax.x_range._property_values['start'] = 0.9
    ax.x_range._property_values['end'] = 2.1
    _title = Title()
    _title.text = 'r_hat'
    ax.title = _title
    ax.xaxis[0].ticker.desired_num_ticks = 3
    return ax