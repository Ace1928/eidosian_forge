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
@pytest.mark.usefixtures('hv_bokeh')
@hv_available
def test_holoviews_pane_reflect_responsive_override(document, comm):
    curve = hv.Curve([1, 2, 3]).opts(responsive=True)
    pane = HoloViews(curve, sizing_mode='fixed')
    row = pane.get_root(document, comm=comm)
    assert row.sizing_mode == 'stretch_both'
    assert pane.sizing_mode == 'fixed'
    pane.sizing_mode = None
    row = pane.get_root(document, comm=comm)
    assert row.sizing_mode == 'stretch_both'
    assert pane.sizing_mode == 'stretch_both'