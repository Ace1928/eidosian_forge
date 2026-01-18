from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def create_multi_index_df():
    arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    return pd.DataFrame(np.random.randn(8, 4), index=arrays)