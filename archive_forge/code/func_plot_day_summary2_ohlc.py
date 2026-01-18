from __future__ import (absolute_import, division, print_function,
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from six.moves import xrange, zip
def plot_day_summary2_ohlc(ax, opens, highs, lows, closes, ticksize=4, colorup='k', colordown='r'):
    """Represent the time, open, high, low, close as a vertical line
    ranging from low to high.  The left tick is the open and the right
    tick is the close.
    *opens*, *highs*, *lows* and *closes* must have the same length.
    NOTE: this code assumes if any value open, high, low, close is
    missing (*-1*) they all are missing

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
    ticksize : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
         the color of the lines where close <  open

    Returns
    -------
    ret : list
        a list of lines added to the axes
    """
    _check_input(opens, highs, lows, closes)
    rangeSegments = [((i, low), (i, high)) for i, low, high in zip(xrange(len(lows)), lows, highs) if low != -1]
    openSegments = [((-ticksize, 0), (0, 0))]
    closeSegments = [((0, 0), (ticksize, 0))]
    offsetsOpen = [(i, open) for i, open in zip(xrange(len(opens)), opens) if open != -1]
    offsetsClose = [(i, close) for i, close in zip(xrange(len(closes)), closes) if close != -1]
    scale = ax.figure.dpi * (1.0 / 72.0)
    tickTransform = Affine2D().scale(scale, 0.0)
    colorup = mcolors.to_rgba(colorup)
    colordown = mcolors.to_rgba(colordown)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close] for open, close in zip(opens, closes) if open != -1 and close != -1]
    useAA = (0,)
    lw = (1,)
    rangeCollection = LineCollection(rangeSegments, colors=colors, linewidths=lw, antialiaseds=useAA)
    openCollection = LineCollection(openSegments, colors=colors, antialiaseds=useAA, linewidths=lw, offsets=offsetsOpen, transOffset=ax.transData)
    openCollection.set_transform(tickTransform)
    closeCollection = LineCollection(closeSegments, colors=colors, antialiaseds=useAA, linewidths=lw, offsets=offsetsClose, transOffset=ax.transData)
    closeCollection.set_transform(tickTransform)
    minpy, maxx = (0, len(rangeSegments))
    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])
    corners = ((minpy, miny), (maxx, maxy))
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(rangeCollection)
    ax.add_collection(openCollection)
    ax.add_collection(closeCollection)
    return (rangeCollection, openCollection, closeCollection)