import datetime as dt
import warnings
import numpy as np
import pytest
from bokeh.models import (
from bokeh.plotting import figure
import panel as pn
from panel.depends import bind
from panel.layout import (
from panel.pane import HoloViews, PaneBase, panel
from panel.tests.util import hv_available, mpl_available
from panel.theme import Native
from panel.util.warnings import PanelDeprecationWarning
from panel.widgets import (
@hv_available
def test_holoviews_widgets_explicit_widget_instance_override():
    hmap = hv.HoloMap({(i, chr(65 + i)): hv.Curve([i]) for i in range(3)}, kdims=['X', 'Y'])
    widget = Select(options=[1, 2, 3], value=3)
    widgets, _ = HoloViews.widgets_from_dimensions(hmap, widget_types={'X': widget})
    assert widgets[0] is widget