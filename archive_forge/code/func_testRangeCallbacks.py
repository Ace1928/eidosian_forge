import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testRangeCallbacks(self):
    range_classes = [RangeXYCallback, RangeXCallback, RangeYCallback]
    xyplots, xystreamss, xycallbacks, xyevents = build_callback_set(RangeXYCallback, ['first', 'second', 'third', 'forth', 'other'], RangeXY, 2)
    xplots, xstreamss, xcallbacks, xevents = build_callback_set(RangeXCallback, ['first', 'second', 'third', 'forth', 'other'], RangeX, 2)
    yplots, ystreamss, ycallbacks, yevents = build_callback_set(RangeYCallback, ['first', 'second', 'third', 'forth', 'other'], RangeY, 2)
    for xystreams in xystreamss:
        self.assertEqual(len(xystreams), 2)
    viewport1 = {'xaxis.range': [1, 4], 'yaxis.range': [-1, 5]}
    for cb_cls in range_classes:
        cb_cls.update_streams_from_property_update('viewport', viewport1, self.fig_dict)
    for xystream, xstream, ystream in zip(xystreamss[0] + xystreamss[1], xstreamss[0] + xstreamss[1], ystreamss[0] + ystreamss[1]):
        assert xystream.x_range == (1, 4)
        assert xystream.y_range == (-1, 5)
        assert xstream.x_range == (1, 4)
        assert ystream.y_range == (-1, 5)
    for xystream, xstream, ystream in zip(xystreamss[2] + xystreamss[3], xstreamss[2] + xstreamss[3], ystreamss[2] + ystreamss[3]):
        assert xystream.x_range is None
        assert xystream.y_range is None
        assert xstream.x_range is None
        assert ystream.y_range is None
    viewport2 = {'xaxis2.range': [2, 5], 'yaxis2.range': [0, 6]}
    for cb_cls in range_classes:
        cb_cls.update_streams_from_property_update('viewport', viewport2, self.fig_dict)
    for xystream, xstream, ystream in zip(xystreamss[2], xstreamss[2], ystreamss[2]):
        assert xystream.x_range == (2, 5)
        assert xystream.y_range == (0, 6)
        assert xstream.x_range == (2, 5)
        assert ystream.y_range == (0, 6)
    viewport3 = {'xaxis3.range': [3, 6], 'yaxis3.range': [1, 7]}
    for cb_cls in range_classes:
        cb_cls.update_streams_from_property_update('viewport', viewport3, self.fig_dict)
    for xystream, xstream, ystream in zip(xystreamss[3], xstreamss[3], ystreamss[3]):
        assert xystream.x_range == (3, 6)
        assert xystream.y_range == (1, 7)
        assert xstream.x_range == (3, 6)
        assert ystream.y_range == (1, 7)
    for xyevent, xevent, yevent in zip(xyevents[4], xevents[4], yevents[4]):
        assert len(xyevent) == 0
        assert len(xevent) == 0
        assert len(yevent) == 0