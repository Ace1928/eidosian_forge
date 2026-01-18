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
def display_multiple_ropes(self, rope, ax, y, linewidth, var_name, selection, plotted, model_name):
    """Display ROPE when more than one interval is provided."""
    for sel in rope.get(var_name, []):
        if all((k in selection and selection[k] == v for k, v in sel.items() if k != 'rope')):
            vals = sel['rope']
            plotted[model_name].append(ax.line(vals, (y + 0.05, y + 0.05), line_width=linewidth * 2, color=[color for _, color in zip(range(3), cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']))][2], line_alpha=0.7))
            return ax