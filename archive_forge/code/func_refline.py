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
def refline(self, *, x=None, y=None, joint=True, marginal=True, color='.5', linestyle='--', **line_kws):
    """Add a reference line(s) to joint and/or marginal axes.

        Parameters
        ----------
        x, y : numeric
            Value(s) to draw the line(s) at.
        joint, marginal : bools
            Whether to add the reference line(s) to the joint/marginal axes.
        color : :mod:`matplotlib color <matplotlib.colors>`
            Specifies the color of the reference line(s).
        linestyle : str
            Specifies the style of the reference line(s).
        line_kws : key, value mappings
            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`
            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``
            is not None.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
    line_kws['color'] = color
    line_kws['linestyle'] = linestyle
    if x is not None:
        if joint:
            self.ax_joint.axvline(x, **line_kws)
        if marginal:
            self.ax_marg_x.axvline(x, **line_kws)
    if y is not None:
        if joint:
            self.ax_joint.axhline(y, **line_kws)
        if marginal:
            self.ax_marg_y.axhline(y, **line_kws)
    return self