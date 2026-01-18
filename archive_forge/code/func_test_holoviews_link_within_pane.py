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
def test_holoviews_link_within_pane(document, comm):
    from bokeh.models.tools import RangeTool
    from holoviews.plotting.links import RangeToolLink
    c1 = hv.Curve([])
    c2 = hv.Curve([])
    RangeToolLink(c1, c2)
    pane = pn.panel(pn.panel(hv.Layout([c1, c2]), backend='bokeh'))
    column = pane.get_root(document, comm=comm)
    assert len(column.children) == 1
    grid_plot = column.children[0]
    assert isinstance(grid_plot, GridPlot)
    assert len(grid_plot.children) == 2
    (p1, _, _), (p2, _, _) = grid_plot.children
    assert isinstance(p1, figure)
    assert isinstance(p2, figure)
    range_tool = grid_plot.select_one({'type': RangeTool})
    assert isinstance(range_tool, RangeTool)
    assert range_tool.x_range == p2.x_range