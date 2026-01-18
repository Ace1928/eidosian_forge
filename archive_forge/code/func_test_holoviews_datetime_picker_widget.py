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
def test_holoviews_datetime_picker_widget(document, comm):
    ds = {'time': [np.datetime64('2000-01-01'), np.datetime64('2000-01-02')], 'x': [0, 1], 'y': [0, 1]}
    viz = hv.Dataset(ds, ['x', 'time'], ['y'])
    layout = pn.panel(viz.to(hv.Scatter, ['x'], ['y']), widgets={'time': pn.widgets.DatetimePicker})
    widget_box = layout[0][1]
    assert isinstance(layout, pn.Row)
    assert isinstance(widget_box, pn.WidgetBox)
    assert isinstance(widget_box[0], pn.widgets.DatetimePicker)