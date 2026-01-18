from __future__ import (absolute_import, division, print_function,
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from six.moves import xrange, zip
def candlestick2_ohlc(ax, opens, highs, lows, closes, width=4, colorup='k', colordown='r', alpha=0.75):
    """Represent the open, close as a bar line and high low range as a
    vertical line.

    NOTE: this code assumes if any value open, low, high, close is
    missing they all are missing


    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    width : float
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency

    Returns
    -------
    ret : tuple
        (lineCollection, barCollection)
    """
    _check_input(opens, highs, lows, closes)
    delta = width / 2.0
    barVerts = [((i - delta, open), (i - delta, close), (i + delta, close), (i + delta, open)) for i, open, close in zip(xrange(len(opens)), opens, closes) if open != -1 and close != -1]
    rangeSegments = [((i, low), (i, high)) for i, low, high in zip(xrange(len(lows)), lows, highs) if low != -1]
    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close] for open, close in zip(opens, closes) if open != -1 and close != -1]
    useAA = (0,)
    lw = (0.5,)
    rangeCollection = LineCollection(rangeSegments, colors=colors, linewidths=lw, antialiaseds=useAA)
    barCollection = PolyCollection(barVerts, facecolors=colors, edgecolors=colors, antialiaseds=useAA, linewidths=lw)
    minx, maxx = (0, len(rangeSegments))
    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])
    corners = ((minx, miny), (maxx, maxy))
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(rangeCollection)
    ax.add_collection(barCollection)
    return (rangeCollection, barCollection)