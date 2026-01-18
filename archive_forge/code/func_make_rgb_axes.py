from types import MethodType
import numpy as np
from .axes_divider import make_axes_locatable, Size
from .mpl_axes import Axes, SimpleAxisArtist
def make_rgb_axes(ax, pad=0.01, axes_class=None, **kwargs):
    """
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes instance to create the RGB Axes in.
    pad : float, optional
        Fraction of the Axes height to pad.
    axes_class : `matplotlib.axes.Axes` or None, optional
        Axes class to use for the R, G, and B Axes. If None, use
        the same class as *ax*.
    **kwargs
        Forwarded to *axes_class* init for the R, G, and B Axes.
    """
    divider = make_axes_locatable(ax)
    pad_size = pad * Size.AxesY(ax)
    xsize = (1 - 2 * pad) / 3 * Size.AxesX(ax)
    ysize = (1 - 2 * pad) / 3 * Size.AxesY(ax)
    divider.set_horizontal([Size.AxesX(ax), pad_size, xsize])
    divider.set_vertical([ysize, pad_size, ysize, pad_size, ysize])
    ax.set_axes_locator(divider.new_locator(0, 0, ny1=-1))
    ax_rgb = []
    if axes_class is None:
        axes_class = type(ax)
    for ny in [4, 2, 0]:
        ax1 = axes_class(ax.get_figure(), ax.get_position(original=True), sharex=ax, sharey=ax, **kwargs)
        locator = divider.new_locator(nx=2, ny=ny)
        ax1.set_axes_locator(locator)
        for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
            t.set_visible(False)
        try:
            for axis in ax1.axis.values():
                axis.major_ticklabels.set_visible(False)
        except AttributeError:
            pass
        ax_rgb.append(ax1)
    fig = ax.get_figure()
    for ax1 in ax_rgb:
        fig.add_axes(ax1)
    return ax_rgb