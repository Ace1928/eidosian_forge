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
def pretty_plot_two_axis(x, y1, y2, xlabel=None, y1label=None, y2label=None, width: float=8, height: float | None=None, dpi=300, **plot_kwargs):
    """Variant of pretty_plot that does a dual axis plot. Adapted from matplotlib
    examples. Makes it easier to create plots with different axes.

    Args:
        x (np.ndarray/list): Data for x-axis.
        y1 (dict/np.ndarray/list): Data for y1 axis (left). If a dict, it will
            be interpreted as a {label: sequence}.
        y2 (dict/np.ndarray/list): Data for y2 axis (right). If a dict, it will
            be interpreted as a {label: sequence}.
        xlabel (str): If not None, this will be the label for the x-axis.
        y1label (str): If not None, this will be the label for the y1-axis.
        y2label (str): If not None, this will be the label for the y2-axis.
        width (float): Width of plot in inches. Defaults to 8in.
        height (float): Height of plot in inches. Defaults to width * golden
            ratio.
        dpi (int): Sets dot per inch for figure. Defaults to 300.
        plot_kwargs: Passthrough kwargs to matplotlib's plot method. E.g.,
            linewidth, etc.

    Returns:
        plt.Axes: matplotlib axes object with properly sized fonts.
    """
    colors = palettable.colorbrewer.diverging.RdYlBu_4.mpl_colors
    c1 = colors[0]
    c2 = colors[-1]
    golden_ratio = (math.sqrt(5) - 1) / 2
    if not height:
        height = int(width * golden_ratio)
    width = 12
    label_size = int(width * 3)
    tick_size = int(width * 2.5)
    styles = ['-', '--', '-.', '.']
    fig, ax1 = plt.subplots()
    fig.set_size_inches((width, height))
    if dpi:
        fig.set_dpi(dpi)
    if isinstance(y1, dict):
        for idx, (key, val) in enumerate(y1.items()):
            ax1.plot(x, val, c=c1, marker='s', ls=styles[idx % len(styles)], label=key, **plot_kwargs)
        ax1.legend(fontsize=label_size)
    else:
        ax1.plot(x, y1, c=c1, marker='s', ls='-', **plot_kwargs)
    if xlabel:
        ax1.set_xlabel(xlabel, fontsize=label_size)
    if y1label:
        ax1.set_ylabel(y1label, color=c1, fontsize=label_size)
    ax1.tick_params('x', labelsize=tick_size)
    ax1.tick_params('y', colors=c1, labelsize=tick_size)
    ax2 = ax1.twinx()
    if isinstance(y2, dict):
        for idx, (key, val) in enumerate(y2.items()):
            ax2.plot(x, val, c=c2, marker='o', ls=styles[idx % len(styles)], label=key)
        ax2.legend(fontsize=label_size)
    else:
        ax2.plot(x, y2, c=c2, marker='o', ls='-')
    if y2label:
        ax2.set_ylabel(y2label, color=c2, fontsize=label_size)
    ax2.tick_params('y', colors=c2, labelsize=tick_size)
    return ax1