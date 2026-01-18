import numpy as np
from bokeh.models import CustomJS, Toolbar
from bokeh.models.tools import RangeTool
from ...core.util import isscalar
from ..links import (
from ..plot import GenericElementPlot, GenericOverlayPlot
@classmethod
def find_link(cls, plot, link=None):
    """
        Searches a GenericElementPlot for a Link.
        """
    registry = Link.registry.items()
    for source in plot.link_sources:
        if link is None:
            links = [l for src, links in registry for l in links if src is source or (src._plot_id is not None and src._plot_id == source._plot_id)]
            if links:
                return (plot, links)
        elif link.target is source or (link.target is not None and link.target._plot_id is not None and (link.target._plot_id == source._plot_id)):
            return (plot, [link])