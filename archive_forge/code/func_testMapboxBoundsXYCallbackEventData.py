import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testMapboxBoundsXYCallbackEventData(self):
    selected_data = {'range': {'mapbox2': [[self.lon_range1[0], self.lat_range1[0]], [self.lon_range1[1], self.lat_range1[1]]]}}
    event_data = BoundsXYCallback.get_event_data_from_property_update('selected_data', selected_data, self.mapbox_fig_dict)
    self.assertEqual(event_data, {'first': {'bounds': None}, 'second': {'bounds': (self.easting_range1[0], self.northing_range1[0], self.easting_range1[1], self.northing_range1[1])}, 'third': {'bounds': None}})