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
def test_holoviews_pane_reflect_responsive_bind_function(document, comm):
    checkbox = Checkbox(value=True)
    curve_fn = lambda responsive: hv.Curve([1, 2, 3]).opts(responsive=responsive)
    pane = panel(bind(curve_fn, responsive=checkbox))
    col = pane.get_root(document, comm=comm)
    assert col.sizing_mode == 'stretch_both'
    checkbox.value = False
    assert col.sizing_mode == 'fixed'