import copy
import math
import warnings
from types import FunctionType
import matplotlib.colors as mpl_colors
import numpy as np
import param
from matplotlib import ticker
from matplotlib.dates import date2num
from matplotlib.image import AxesImage
from packaging.version import Version
from ...core import (
from ...core.options import Keywords, abbreviated_exception
from ...element import Graph, Path
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_range_key, process_cmap
from .plot import MPLPlot, mpl_rc_context
from .util import EqHistNormalize, mpl_version, validate, wrap_formatter
def render_artists(self, element, ranges, style, ax):
    plot_data, plot_kwargs, axis_kwargs = self.get_data(element, ranges, style)
    legend = plot_kwargs.pop('cat_legend', None)
    with abbreviated_exception():
        handles = self.init_artists(ax, plot_data, plot_kwargs)
    if legend and 'artist' in handles and hasattr(handles['artist'], 'legend_elements'):
        legend_handles, _ = handles['artist'].legend_elements()
        leg = ax.legend(legend_handles, legend['factors'], title=legend['title'], **self._legend_opts)
        ax.add_artist(leg)
    return (handles, axis_kwargs)