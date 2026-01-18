from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_change_selection():
    widget = QgridWidget(df=create_large_df(size=10))
    event_history = init_event_history('selection_changed', widget=widget)
    widget._handle_qgrid_msg_helper({'type': 'change_selection', 'rows': [5]})
    assert widget._selected_rows == [5]
    widget._handle_qgrid_msg_helper({'type': 'change_selection', 'rows': [7, 8]})
    assert widget._selected_rows == [7, 8]
    widget.change_selection([3, 5, 6])
    assert widget._selected_rows == [3, 5, 6]
    widget.change_selection()
    assert widget._selected_rows == []
    assert event_history == [{'name': 'selection_changed', 'old': [], 'new': [5], 'source': 'gui'}, {'name': 'selection_changed', 'old': [5], 'new': [7, 8], 'source': 'gui'}, {'name': 'selection_changed', 'old': [7, 8], 'new': [3, 5, 6], 'source': 'api'}, {'name': 'selection_changed', 'old': [3, 5, 6], 'new': [], 'source': 'api'}]