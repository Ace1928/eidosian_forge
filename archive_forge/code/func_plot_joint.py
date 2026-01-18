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
def plot_joint(self, func, **kwargs):
    """Draw a bivariate plot on the joint axes of the grid.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should accept ``x`` and ``y``. Otherwise,
            it must accept ``x`` and ``y`` vectors of data as the first two
            positional arguments, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, the function must
            accept ``hue`` as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
    kwargs = kwargs.copy()
    if str(func.__module__).startswith('seaborn'):
        kwargs['ax'] = self.ax_joint
    else:
        plt.sca(self.ax_joint)
    if self.hue is not None:
        kwargs['hue'] = self.hue
        self._inject_kwargs(func, kwargs, self._hue_params)
    if str(func.__module__).startswith('seaborn'):
        func(x=self.x, y=self.y, **kwargs)
    else:
        func(self.x, self.y, **kwargs)
    return self