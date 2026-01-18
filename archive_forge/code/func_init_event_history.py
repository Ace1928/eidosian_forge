from qgrid import QgridWidget, set_defaults, show_grid, on as qgrid_on
from traitlets import All
import numpy as np
import pandas as pd
import json
def init_event_history(event_names, widget=None):
    event_history = []

    def on_change(event, qgrid_widget):
        event_history.append(event)
    if widget is not None:
        widget.on(event_names, on_change)
    else:
        qgrid_on(event_names, on_change)
    return event_history