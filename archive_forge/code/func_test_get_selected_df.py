from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_get_selected_df():
    sample_df = create_df()
    selected_rows = [1, 3]
    view = QgridWidget(df=sample_df)
    view._handle_qgrid_msg_helper({'rows': selected_rows, 'type': 'change_selection'})
    selected_df = view.get_selected_df()
    assert len(selected_df) == 2
    assert sample_df.iloc[selected_rows].equals(selected_df)