from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
@add_fig_kwargs
def scatter_hist(self, ax: plt.Axes=None, **kwargs):
    """
        Scatter plot + histogram.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            plt.Figure: matplotlib figure
        """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax, fig = get_ax_fig(ax=ax)
    x = np.asarray(self.get_values('cpu_time'))
    y = np.asarray(self.get_values('wall_time'))
    axScatter = plt.subplot(1, 1, 1)
    axScatter.scatter(x, y)
    axScatter.set_aspect('auto')
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes('top', 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes('right', 1.2, pad=0.1, sharey=axScatter)
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')
    axHistx.set_yticks([0, 50, 100])
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
        for tl in axHisty.get_yticklabels():
            tl.set_visible(False)
            axHisty.set_xticks([0, 50, 100])
    return fig