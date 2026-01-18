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
def test_holoviews_widgets_update_plot(document, comm):
    hmap = hv.HoloMap({(i, chr(65 + i)): hv.Curve([i]) for i in range(3)}, kdims=['X', 'Y'])
    hv_pane = HoloViews(hmap, backend='bokeh')
    layout = hv_pane.get_root(document, comm)
    cds = layout.children[0].select_one({'type': ColumnDataSource})
    assert cds.data['y'] == np.array([0])
    hv_pane.widget_box[0].value = 1
    hv_pane.widget_box[1].value = chr(65 + 1)
    assert cds.data['y'] == np.array([1])