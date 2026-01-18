import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def filter_plotters_list(plotters, plot_kind):
    """Cut list of plotters so that it is at most of length "plot.max_subplots"."""
    max_plots = rcParams['plot.max_subplots']
    max_plots = len(plotters) if max_plots is None else max_plots
    if len(plotters) > max_plots:
        warnings.warn("rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number of variables to plot ({len_plotters}) in {plot_kind}, generating only {max_plots} plots".format(max_plots=max_plots, len_plotters=len(plotters), plot_kind=plot_kind), UserWarning)
        return plotters[:max_plots]
    return plotters