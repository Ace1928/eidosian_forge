from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_integer_index_filter():
    view = QgridWidget(df=create_df())
    view._handle_qgrid_msg_helper({'field': 'index', 'filter_info': {'field': 'index', 'max': None, 'min': 2, 'type': 'slider'}, 'type': 'change_filter'})
    filtered_df = view.get_changed_df()
    assert len(filtered_df) == 2