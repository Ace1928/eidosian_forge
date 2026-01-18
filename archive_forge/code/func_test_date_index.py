from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_date_index():
    df = create_df()
    df.set_index('Date', inplace=True)
    view = QgridWidget(df=df)
    view._handle_qgrid_msg_helper({'type': 'change_filter', 'field': 'A', 'filter_info': {'field': 'A', 'type': 'slider', 'min': 2, 'max': 3}})