import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def plot_confound(tseries, figsize, name, units=None, series_tr=None, normalize=False):
    """
    A helper function to plot :abbr:`fMRI (functional MRI)` confounds.

    """
    import matplotlib
    matplotlib.use(config.get('execution', 'matplotlib_backend'))
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
    import seaborn as sns
    fig = plt.Figure(figsize=figsize)
    FigureCanvas(fig)
    grid = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.025)
    grid.update(hspace=1.0, right=0.95, left=0.1, bottom=0.2)
    ax = fig.add_subplot(grid[0, :-1])
    if normalize and series_tr is not None:
        tseries /= series_tr
    ax.plot(tseries)
    ax.set_xlim((0, len(tseries)))
    ylabel = name
    if units is not None:
        ylabel += (' speed [{}/s]' if normalize else ' [{}]').format(units)
    ax.set_ylabel(ylabel)
    xlabel = 'Frame #'
    if series_tr is not None:
        xlabel = 'Frame # ({} sec TR)'.format(series_tr)
    ax.set_xlabel(xlabel)
    ylim = ax.get_ylim()
    ax = fig.add_subplot(grid[0, -1])
    sns.distplot(tseries, vertical=True, ax=ax)
    ax.set_xlabel('Frames')
    ax.set_ylim(ylim)
    ax.set_yticklabels([])
    return fig