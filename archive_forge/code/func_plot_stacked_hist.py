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
def plot_stacked_hist(self, key='wall_time', nmax=5, ax: plt.Axes=None, **kwargs):
    """
        Plot stacked histogram of the different timers.

        Args:
            key: Keyword used to extract data from the timers. Only the first `nmax`
                sections with largest value are show.
            nmax: Maximum number of sections to show. Other entries are grouped together
                in the `others` section.
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            `matplotlib` figure
        """
    ax, fig = get_ax_fig(ax=ax)
    mpi_rank = '0'
    timers = self.timers(mpi_rank=mpi_rank)
    n = len(timers)
    names, values = ([], [])
    rest = np.zeros(n)
    for idx, sec_name in enumerate(self.section_names(ordkey=key)):
        sections = self.get_sections(sec_name)
        sec_vals = np.asarray([s.__dict__[key] for s in sections])
        if idx < nmax:
            names.append(sec_name)
            values.append(sec_vals)
        else:
            rest += sec_vals
    names.append(f'others (nmax={nmax!r})')
    values.append(rest)
    ind = np.arange(n)
    width = 0.35
    colors = nmax * ['r', 'g', 'b', 'c', 'k', 'y', 'm']
    bars = []
    bottom = np.zeros(n)
    for idx, vals in enumerate(values):
        color = colors[idx]
        bar_ = ax.bar(ind, vals, width, color=color, bottom=bottom)
        bars.append(bar_)
        bottom += vals
    ax.set_ylabel(key)
    ax.set_title(f'Stacked histogram with the {nmax} most important sections')
    ticks = ind + width / 2.0
    labels = [f'MPI={timer.mpi_nprocs}, OMP={timer.omp_nthreads}' for timer in timers]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend([bar_[0] for bar_ in bars], names, loc='best')
    return fig