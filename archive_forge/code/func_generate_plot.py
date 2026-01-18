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
def generate_plot(self, key, ranges, element=None, is_geo=False):
    if element is None:
        element = self._get_frame(key)
    items = [] if element is None else list(element.data.items())
    plot_opts = self.lookup_options(element, 'plot').options
    inherited = self._traverse_options(element, 'plot', self._propagate_options, defaults=False)
    plot_opts.update(**{k: v[0] for k, v in inherited.items() if k not in plot_opts})
    self.param.update(**plot_opts)
    ranges = self.compute_ranges(self.hmap, key, ranges)
    figure = None
    for _, el in items:
        if isinstance(el, Tiles):
            is_geo = True
            break
    for okey, subplot in self.subplots.items():
        if element is not None and subplot.drawn:
            idx, spec, exact = self._match_subplot(okey, subplot, items, element)
            if idx is not None:
                _, el = items.pop(idx)
            else:
                el = None
        else:
            el = None
        subplot.param.update(**plot_opts)
        fig = subplot.generate_plot(key, ranges, el, is_geo=is_geo)
        if figure is None:
            figure = fig
        else:
            merge_figure(figure, fig)
    layout = self.init_layout(key, element, ranges, is_geo=is_geo)
    merge_layout(figure['layout'], layout)
    self.drawn = True
    self.handles['fig'] = figure
    return figure