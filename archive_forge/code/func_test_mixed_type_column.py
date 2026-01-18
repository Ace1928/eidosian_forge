from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_mixed_type_column():
    df = pd.DataFrame({'A': [1.2, 'xy', 4], 'B': [3, 4, 5]})
    df = df.set_index(pd.Index(['yz', 7, 3.2]))
    view = QgridWidget(df=df)
    view._handle_qgrid_msg_helper({'type': 'change_sort', 'sort_field': 'A', 'sort_ascending': True})
    view._handle_qgrid_msg_helper({'type': 'show_filter_dropdown', 'field': 'A', 'search_val': None})