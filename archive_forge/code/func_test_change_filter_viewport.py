from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_change_filter_viewport():
    widget = QgridWidget(df=create_large_df())
    event_history = init_event_history(All)
    widget._handle_qgrid_msg_helper({'type': 'show_filter_dropdown', 'field': 'B (as str)', 'search_val': None})
    widget._handle_qgrid_msg_helper({'type': 'change_filter_viewport', 'field': 'B (as str)', 'top': 556, 'bottom': 568})
    widget._handle_qgrid_msg_helper({'type': 'change_filter_viewport', 'field': 'B (as str)', 'top': 302, 'bottom': 314})
    assert event_history == [{'name': 'filter_dropdown_shown', 'column': 'B (as str)'}, {'name': 'text_filter_viewport_changed', 'column': 'B (as str)', 'old': (0, 200), 'new': (556, 568)}, {'name': 'text_filter_viewport_changed', 'column': 'B (as str)', 'old': (556, 568), 'new': (302, 314)}]