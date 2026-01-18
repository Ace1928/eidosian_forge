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
def test_holoviews_not_linked_axes(document, comm):
    c1 = hv.Curve([1, 2, 3])
    c2 = hv.Curve([1, 2, 3])
    layout = Row(HoloViews(c1, backend='bokeh'), HoloViews(c2, backend='bokeh', linked_axes=False))
    row_model = layout.get_root(document, comm=comm)
    p1, p2 = row_model.select({'type': figure})
    assert p1.x_range is not p2.x_range
    assert p1.y_range is not p2.y_range