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
def test_holoviews_property_override(document, comm):
    c1 = hv.Curve([])
    pane = pn.panel(c1, backend='bokeh', styles={'background': 'red'}, css_classes=['test_class'])
    model = pane.get_root(document, comm=comm)
    assert model.styles['background'] == 'red'
    assert model.children[0].css_classes == ['test_class']