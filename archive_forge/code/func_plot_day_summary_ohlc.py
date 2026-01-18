from __future__ import (absolute_import, division, print_function,
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from six.moves import xrange, zip
def plot_day_summary_ohlc(ax, quotes, ticksize=3, colorup='k', colordown='r'):
    """Plots day summary

        Represent the time, open, high, low, close as a vertical line
        ranging from low to high.  The left tick is the open and the right
        tick is the close.



    Parameters
    ----------
    ax : `Axes`
        an `Axes` instance to plot to
    quotes : sequence of (time, open, high, low, close, ...) sequences
        data to plot.  time must be in float date format - see date2num
    ticksize : int
        open/close tick marker in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open

    Returns
    -------
    lines : list
        list of tuples of the lines added (one tuple per quote)
    """
    return _plot_day_summary(ax, quotes, ticksize=ticksize, colorup=colorup, colordown=colordown, ochl=False)