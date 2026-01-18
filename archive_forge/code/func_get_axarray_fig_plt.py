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
def get_axarray_fig_plt(ax_array, nrows=1, ncols=1, sharex: bool=False, sharey: bool=False, squeeze: bool=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
    """Helper function used in plot functions that accept an optional array of Axes
    as argument. If ax_array is None, we build the `matplotlib` figure and
    create the array of Axes by calling plt.subplots else we return the
    current active figure.

    Returns:
        ax: Array of Axes objects
        figure: matplotlib figure
        plt: matplotlib pyplot module.
    """
    if ax_array is None:
        fig, ax_array = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=squeeze, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)
    else:
        fig = plt.gcf()
        ax_array = np.reshape(np.array(ax_array), (nrows, ncols))
        if squeeze:
            if ax_array.size == 1:
                ax_array = ax_array[0]
            elif any((s == 1 for s in ax_array.shape)):
                ax_array = ax_array.ravel()
    return (ax_array, fig, plt)