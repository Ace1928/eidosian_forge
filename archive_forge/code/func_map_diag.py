from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
def map_diag(self, func, **kwargs):
    """Plot with a univariate function on each diagonal subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take an x array as a positional argument and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
    if self.diag_axes is None:
        diag_vars = []
        diag_axes = []
        for i, y_var in enumerate(self.y_vars):
            for j, x_var in enumerate(self.x_vars):
                if x_var == y_var:
                    diag_vars.append(x_var)
                    ax = self.axes[i, j]
                    diag_ax = ax.twinx()
                    diag_ax.set_axis_off()
                    diag_axes.append(diag_ax)
                    if not plt.rcParams.get('ytick.left', True):
                        for tick in ax.yaxis.majorTicks:
                            tick.tick1line.set_visible(False)
                    if self._corner:
                        ax.yaxis.set_visible(False)
                        if self._despine:
                            utils.despine(ax=ax, left=True)
        if self.diag_sharey and diag_axes:
            for ax in diag_axes[1:]:
                share_axis(diag_axes[0], ax, 'y')
        self.diag_vars = diag_vars
        self.diag_axes = diag_axes
    if 'hue' not in signature(func).parameters:
        return self._map_diag_iter_hue(func, **kwargs)
    for var, ax in zip(self.diag_vars, self.diag_axes):
        plot_kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            plot_kwargs['ax'] = ax
        else:
            plt.sca(ax)
        vector = self.data[var]
        if self._hue_var is not None:
            hue = self.data[self._hue_var]
        else:
            hue = None
        if self._dropna:
            not_na = vector.notna()
            if hue is not None:
                not_na &= hue.notna()
            vector = vector[not_na]
            if hue is not None:
                hue = hue[not_na]
        plot_kwargs.setdefault('hue', hue)
        plot_kwargs.setdefault('hue_order', self._hue_order)
        plot_kwargs.setdefault('palette', self._orig_palette)
        func(x=vector, **plot_kwargs)
        ax.legend_ = None
    self._add_axis_labels()
    return self