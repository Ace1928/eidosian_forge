import numpy as np
import holoviews as hv
from holoviews import opts
from . import get_aliases, all_original_names, palette, cm
from .sineramp import sineramp
def swatch(name, cmap=None, bounds=None, array=array, **kwargs):
    """Show a color swatch for a colormap using matplotlib or bokeh via holoviews.
    Colormaps can be selected by `name`, including those in Colorcet
    along with any standard Bokeh palette or named Matplotlib colormap.

    Custom colormaps can be visualized by passing an explicit
    list of colors (for Bokeh) or the colormap object (for Matplotlib) to `cmap`.

    HoloViews options for either backend can be passed in as kwargs,
    so that you can customize the width, height, etc. of the swatch.

    The `bounds` and `array` arguments allow you to customize the
    portion of the colormap to show and how many samples to take
    from it; see the source code and hv.Image documentation for
    details.
    """
    title = name if cmap else get_aliases(name)
    if bounds is None:
        bounds = (0, 0, 256, 1)
    if type(cmap) is tuple:
        cmap = list(cmap)
    plot = hv.Image(array, bounds=bounds, group=title)
    backends = hv.Store.loaded_backends()
    if 'bokeh' in backends:
        width = kwargs.pop('width', 900)
        height = kwargs.pop('height', 100)
        plot.opts(opts.Image(backend='bokeh', width=width, height=height, toolbar='above', default_tools=['xwheel_zoom', 'xpan', 'save', 'reset'], cmap=cmap or palette[name]))
    if 'matplotlib' in backends:
        aspect = kwargs.pop('aspect', 15)
        fig_size = kwargs.pop('fig_size', 350)
        plot.opts(opts.Image(backend='matplotlib', aspect=aspect, fig_size=fig_size, cmap=cmap or cm[name]))
    return plot.opts(opts.Image(xaxis=None, yaxis=None), opts.Image(**kwargs))