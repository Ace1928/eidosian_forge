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
def plot_marginals(self, func, **kwargs):
    """Draw univariate plots on each marginal axes.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should  accept ``x`` and ``y`` and plot
            when only one of them is defined. Otherwise, it must accept a vector
            of data as the first positional argument and determine its orientation
            using the ``vertical`` parameter, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, it must accept ``hue``
            as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
    seaborn_func = str(func.__module__).startswith('seaborn') and (not func.__name__ == 'distplot')
    func_params = signature(func).parameters
    kwargs = kwargs.copy()
    if self.hue is not None:
        kwargs['hue'] = self.hue
        self._inject_kwargs(func, kwargs, self._hue_params)
    if 'legend' in func_params:
        kwargs.setdefault('legend', False)
    if 'orientation' in func_params:
        orient_kw_x = {'orientation': 'vertical'}
        orient_kw_y = {'orientation': 'horizontal'}
    elif 'vertical' in func_params:
        orient_kw_x = {'vertical': False}
        orient_kw_y = {'vertical': True}
    if seaborn_func:
        func(x=self.x, ax=self.ax_marg_x, **kwargs)
    else:
        plt.sca(self.ax_marg_x)
        func(self.x, **orient_kw_x, **kwargs)
    if seaborn_func:
        func(y=self.y, ax=self.ax_marg_y, **kwargs)
    else:
        plt.sca(self.ax_marg_y)
        func(self.y, **orient_kw_y, **kwargs)
    self.ax_marg_x.yaxis.get_label().set_visible(False)
    self.ax_marg_y.xaxis.get_label().set_visible(False)
    return self