from __future__ import annotations
import importlib
import math
from functools import wraps
from string import ascii_letters
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.diverging
from matplotlib import cm, colors
from pymatgen.core import Element
def pretty_polyfit_plot(x: ArrayLike, y: ArrayLike, deg: int=1, xlabel=None, ylabel=None, **kwargs):
    """Convenience method to plot data with trend lines based on polynomial fit.

    Args:
        x: Sequence of x data.
        y: Sequence of y data.
        deg (int): Degree of polynomial. Defaults to 1.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        kwargs: Keyword args passed to pretty_plot.

    Returns:
        matplotlib.pyplot object.
    """
    ax = pretty_plot(**kwargs)
    pp = np.polyfit(x, y, deg)
    xp = np.linspace(min(x), max(x), 200)
    ax.plot(xp, np.polyval(pp, xp), 'k--', x, y, 'o')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax