from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def assert_widget_vals_a(widget):
    assert not widget.show_toolbar
    assert widget.precision == 4
    assert widget.grid_options == fake_grid_options_a