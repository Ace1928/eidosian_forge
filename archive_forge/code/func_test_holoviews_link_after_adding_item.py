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
def test_holoviews_link_after_adding_item(document, comm):
    from bokeh.models.tools import RangeTool
    from holoviews.plotting.links import RangeToolLink
    c1 = hv.Curve([])
    c2 = hv.Curve([])
    RangeToolLink(c1, c2)
    layout = Row(pn.panel(c1, backend='bokeh'))
    row = layout.get_root(document, comm=comm)
    assert len(row.children) == 1
    p1, = row.children
    assert isinstance(p1, figure)
    range_tool = row.select_one({'type': RangeTool})
    assert range_tool is None
    layout.append(pn.panel(c2, backend='bokeh'))
    _, p2 = row.children
    assert isinstance(p2, figure)
    range_tool = row.select_one({'type': RangeTool})
    assert isinstance(range_tool, RangeTool)
    assert range_tool.x_range == p2.x_range