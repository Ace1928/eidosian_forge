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
def test_holoviews_with_widgets(document, comm):
    hmap = hv.HoloMap({(i, chr(65 + i)): hv.Curve([i]) for i in range(3)}, kdims=['X', 'Y'])
    hv_pane = HoloViews(hmap)
    layout = hv_pane.get_root(document, comm)
    model = layout.children[0]
    assert len(hv_pane.widget_box.objects) == 2
    assert hv_pane.widget_box.objects[0].name == 'X'
    assert hv_pane.widget_box.objects[1].name == 'Y'
    assert hv_pane._models[layout.ref['id']][0] is model
    hmap = hv.HoloMap({(i, chr(65 + i)): hv.Curve([i]) for i in range(3)}, kdims=['A', 'B'])
    hv_pane.object = hmap
    assert len(hv_pane.widget_box.objects) == 2
    assert hv_pane.widget_box.objects[0].name == 'A'
    assert hv_pane.widget_box.objects[1].name == 'B'