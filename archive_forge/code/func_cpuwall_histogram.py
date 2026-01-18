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
def cpuwall_histogram(self, ax: plt.Axes=None, **kwargs):
    """
        Plot histogram with cpu- and wall-time on axis `ax`.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            plt.Figure: matplotlib figure
        """
    ax, fig = get_ax_fig(ax=ax)
    ind = np.arange(len(self.sections))
    width = 0.35
    cpu_times = self.get_values('cpu_time')
    rects1 = plt.bar(ind, cpu_times, width, color='r')
    wall_times = self.get_values('wall_time')
    rects2 = plt.bar(ind + width, wall_times, width, color='y')
    ax.set_ylabel('Time (s)')
    ticks = self.get_values('name')
    ax.set_xticks(ind + width, ticks)
    ax.legend((rects1[0], rects2[0]), ('CPU', 'Wall'), loc='best')
    return fig