import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
def get_color_opts(self, eldim, element, ranges, style):
    opts = {}
    dim_name = dim_range_key(eldim)
    if self.colorbar:
        opts['colorbar'] = dict(**self.colorbar_opts)
        if 'title' not in opts['colorbar']:
            if isinstance(eldim, dim):
                title = str(eldim)
                if eldim.ops:
                    pass
                elif title.startswith("dim('") and title.endswith("')"):
                    title = title[5:-2]
                else:
                    title = title[1:-1]
            else:
                title = eldim.pprint_label
            opts['colorbar']['title'] = title
        opts['showscale'] = True
    else:
        opts['showscale'] = False
    if eldim:
        auto = False
        if util.isfinite(self.clim).all():
            cmin, cmax = self.clim
        elif dim_name in ranges:
            if self.clim_percentile and 'robust' in ranges[dim_name]:
                low, high = ranges[dim_name]['robust']
            else:
                cmin, cmax = ranges[dim_name]['combined']
        elif isinstance(eldim, dim):
            cmin, cmax = (np.nan, np.nan)
            auto = True
        else:
            cmin, cmax = element.range(dim_name)
        if self.symmetric:
            cabs = np.abs([cmin, cmax])
            cmin, cmax = (-cabs.max(), cabs.max())
    else:
        auto = True
        cmin, cmax = (None, None)
    cmap = style.pop('cmap', 'viridis')
    colorscale = get_colorscale(cmap, self.color_levels, cmin, cmax)
    if isinstance(colorscale, list) and len(colorscale) > 255:
        last_clr_pair = colorscale[-1]
        step = int(np.ceil(len(colorscale) / 255))
        colorscale = colorscale[0::step]
        colorscale[-1] = last_clr_pair
    if cmin is not None:
        opts['cmin'] = cmin
    if cmax is not None:
        opts['cmax'] = cmax
    opts['cauto'] = auto
    opts['colorscale'] = colorscale
    return opts