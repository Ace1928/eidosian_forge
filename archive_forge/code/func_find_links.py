from __future__ import annotations
import sys
from collections import defaultdict
from functools import partial
from typing import (
import param
from bokeh.models import Range1d, Spacer as _BkSpacer
from bokeh.themes.theme import Theme
from packaging.version import Version
from param.parameterized import register_reference_transform
from param.reactive import bind
from ..io import state, unlocked
from ..layout import (
from ..viewable import Layoutable, Viewable
from ..widgets import Player
from .base import PaneBase, RerenderError, panel
from .plot import Bokeh, Matplotlib
from .plotly import Plotly
def find_links(root_view, root_model):
    """
    Traverses the supplied Viewable searching for Links between any
    HoloViews based panes.
    """
    hv_views = root_view.select(HoloViews)
    root_plots = [plot for view in hv_views for plot, _ in view._plots.values() if getattr(plot, 'root', None) is root_model]
    if not len(root_plots) > 1:
        return
    try:
        try:
            from holoviews.plotting.bokeh.links import LinkCallback
        except Exception:
            from holoviews.plotting.bokeh.callbacks import LinkCallback
        from holoviews.plotting.links import Link
    except Exception:
        return
    plots = [(plot, root_plot) for root_plot in root_plots for plot in root_plot.traverse(lambda x: x, [is_bokeh_element_plot])]
    potentials = [(LinkCallback.find_link(plot), root_plot) for plot, root_plot in plots]
    source_links = [p for p in potentials if p[0] is not None]
    found = []
    for (plot, links), root_plot in source_links:
        for link in links:
            if link.target is None:
                found.append((link, plot, None))
                continue
            potentials = [LinkCallback.find_link(plot, link) for plot, inner_root in plots if inner_root is not root_plot]
            tgt_links = [p for p in potentials if p is not None]
            if tgt_links:
                found.append((link, plot, tgt_links[0][0]))
    new_found = set(found) - root_view._found_links
    callbacks = []
    for link, src_plot, tgt_plot in new_found:
        cb = Link._callbacks['bokeh'][type(link)]
        if src_plot is None or (getattr(link, '_requires_target', False) and tgt_plot is None):
            continue
        callbacks.append(cb(root_model, link, src_plot, tgt_plot))
    root_view._found_links.update(new_found)
    return callbacks