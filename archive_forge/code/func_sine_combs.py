import numpy as np
import holoviews as hv
from holoviews import opts
from . import get_aliases, all_original_names, palette, cm
from .sineramp import sineramp
def sine_combs(*args, group=None, not_group=None, only_aliased=False, cols=1, **kwargs):
    """Show sine_combs for given names or names in group"""
    args = args or all_original_names(group=group, not_group=not_group, only_aliased=only_aliased)
    images = [sine_comb(arg, **kwargs) if isinstance(arg, str) else sine_comb(*arg, **kwargs) for arg in args]
    plot = hv.Layout(images).opts(transpose=True).cols(int(np.ceil(len(images) * 1.0 / cols)))
    backends = hv.Store.loaded_backends()
    if 'matplotlib' in backends:
        plot.opts(opts.Layout(backend='matplotlib', sublabel_format=None, fig_size=kwargs.get('fig_size', 200)))
    return plot