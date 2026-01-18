import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testMapboxRangeXYCallbackEventData(self):
    relayout_data = {'mapbox._derived': {'coordinates': self.mapbox_coords1}, 'mapbox3._derived': {'coordinates': self.mapbox_coords2}}
    event_data = RangeXYCallback.get_event_data_from_property_update('relayout_data', relayout_data, self.mapbox_fig_dict)
    self.assertEqual(event_data, {'first': {'x_range': self.easting_range1, 'y_range': self.northing_range1}, 'third': {'x_range': self.easting_range2, 'y_range': self.northing_range2}})