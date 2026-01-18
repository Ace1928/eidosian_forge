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
def test_holoviews_dynamic_widgets_with_unit_updates_plot(document, comm):

    def function(f):
        return hv.Curve((x, np.sin(f * x)))
    x = np.linspace(0, 10)
    factor = hv.Dimension('factor', unit='m', values=[1, 2, 3, 4, 5])
    dmap = hv.DynamicMap(function, kdims=factor)
    hv_pane = HoloViews(dmap, backend='bokeh')
    layout = hv_pane.get_root(document, comm)
    cds = layout.children[0].select_one({'type': ColumnDataSource})
    np.testing.assert_array_equal(cds.data['y'], np.sin(x))
    hv_pane.widget_box[0].value = 3
    np.testing.assert_array_equal(cds.data['y'], np.sin(3 * x))