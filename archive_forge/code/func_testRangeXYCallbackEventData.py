import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testRangeXYCallbackEventData(self):
    for viewport in [{'xaxis.range': [1, 4], 'yaxis.range': [-1, 5]}, {'xaxis.range[0]': 1, 'xaxis.range[1]': 4, 'yaxis.range[0]': -1, 'yaxis.range[1]': 5}]:
        event_data = RangeXYCallback.get_event_data_from_property_update('viewport', viewport, self.fig_dict)
        self.assertEqual(event_data, {'first': {'x_range': (1, 4), 'y_range': (-1, 5)}, 'second': {'x_range': (1, 4), 'y_range': (-1, 5)}})