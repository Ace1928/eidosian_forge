from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def test_multi_interval_index():
    df = create_interval_index_df()
    df['A'] = np.array([3] * 1000, dtype='int32')
    df.set_index(['time', 'time_bin'], inplace=True)
    show_grid(df)