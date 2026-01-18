import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testCallbackClassInstanceTracking(self):
    plot1 = mock_plot()
    plot2 = mock_plot()
    plot3 = mock_plot()
    rangexy_cb = RangeXYCallback(plot1, [], None)
    self.assertIn(plot1.trace_uid, RangeXYCallback.instances)
    self.assertIs(rangexy_cb, RangeXYCallback.instances[plot1.trace_uid])
    boundsxy_cb = BoundsXYCallback(plot2, [], None)
    self.assertIn(plot2.trace_uid, BoundsXYCallback.instances)
    self.assertIs(boundsxy_cb, BoundsXYCallback.instances[plot2.trace_uid])
    selection1d_cb = Selection1DCallback(plot3, [], None)
    self.assertIn(plot3.trace_uid, Selection1DCallback.instances)
    self.assertIs(selection1d_cb, Selection1DCallback.instances[plot3.trace_uid])
    self.assertNotIn(plot1.trace_uid, BoundsXYCallback.instances)
    self.assertNotIn(plot1.trace_uid, Selection1DCallback.instances)
    self.assertNotIn(plot2.trace_uid, RangeXYCallback.instances)
    self.assertNotIn(plot2.trace_uid, Selection1DCallback.instances)
    self.assertNotIn(plot3.trace_uid, RangeXYCallback.instances)
    self.assertNotIn(plot3.trace_uid, BoundsXYCallback.instances)