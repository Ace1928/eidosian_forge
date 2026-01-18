from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_edit_number():
    old_val = 3
    view = QgridWidget(df=create_df())
    for idx in range(-10, 10, 1):
        check_edit_success(view, 'D', 2, old_val, old_val, idx, idx)
        old_val = idx