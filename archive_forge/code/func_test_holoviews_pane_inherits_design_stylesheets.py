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
@pytest.mark.usefixtures('hv_plotly')
@hv_available
@plotly_available
def test_holoviews_pane_inherits_design_stylesheets(document, comm):
    curve = hv.Curve([1, 2, 3]).opts(responsive=True, backend='plotly')
    pane = HoloViews(curve, backend='plotly')
    row = pane.get_root(document, comm=comm)
    Native().apply(pane, row)
    plotly_model = row.children[0]
    assert len(plotly_model.stylesheets) == 5