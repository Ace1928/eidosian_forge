from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def create_df():
    return pd.DataFrame({'A': 1.0, 'Date': pd.Timestamp('20130102'), 'C': pd.Series(1, index=list(range(4)), dtype='float32'), 'D': np.array([3] * 4, dtype='int32'), 'E': pd.Categorical(['test', 'train', 'foo', 'bar']), 'F': ['foo', 'bar', 'buzz', 'fox']})