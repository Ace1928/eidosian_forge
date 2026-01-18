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
def is_bokeh_element_plot(plot):
    """
    Checks whether plotting instance is a HoloViews ElementPlot rendered
    with the bokeh backend.
    """
    from holoviews.plotting.plot import GenericElementPlot, GenericOverlayPlot, Plot
    if not isinstance(plot, Plot):
        return False
    return plot.renderer.backend == 'bokeh' and isinstance(plot, GenericElementPlot) and (not isinstance(plot, GenericOverlayPlot))