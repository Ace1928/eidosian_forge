import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testBoundsYCallbackEventData(self):
    selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
    event_data = BoundsYCallback.get_event_data_from_property_update('selected_data', selected_data1, self.fig_dict)
    self.assertEqual(event_data, {'first': {'boundsy': (-1, 5)}, 'second': {'boundsy': (-1, 5)}, 'third': {'boundsy': None}, 'forth': {'boundsy': None}})