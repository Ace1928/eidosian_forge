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
def test_holoviews_updates_widgets(document, comm):
    hmap = hv.HoloMap({(i, chr(65 + i)): hv.Curve([i]) for i in range(3)}, kdims=['X', 'Y'])
    hv_pane = HoloViews(hmap)
    layout = hv_pane.get_root(document, comm)
    hv_pane.widgets = {'X': Select}
    assert isinstance(hv_pane.widget_box[0], Select)
    assert isinstance(layout.children[1].children[1], BkSelect)
    hv_pane.widgets = {'X': DiscreteSlider}
    assert isinstance(hv_pane.widget_box[0], DiscreteSlider)
    assert isinstance(layout.children[1].children[0], BkColumn)
    assert isinstance(layout.children[1].children[0].children[1], BkSlider)