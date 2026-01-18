import numpy as np
import holoviews as hv
from holoviews import opts
from . import get_aliases, all_original_names, palette, cm
from .sineramp import sineramp
def sine_comb(name, cmap=None, **kwargs):
    """Show sine_comb using matplotlib or bokeh via holoviews"""
    title = name if cmap else get_aliases(name)
    plot = hv.Image(sine, group=title)
    backends = hv.Store.loaded_backends()
    if 'bokeh' in backends:
        plot.opts(opts.Image(backend='bokeh', width=400, height=150, toolbar='above', cmap=cmap or palette[name]))
    if 'matplotlib' in backends:
        plot.opts(opts.Image(backend='matplotlib', aspect=3, fig_size=200, cmap=cmap or cm[name]))
    return plot.opts(opts.Image(xaxis=None, yaxis=None), opts.Image(**kwargs))