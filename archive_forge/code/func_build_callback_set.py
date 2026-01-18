import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def build_callback_set(callback_cls, trace_uids, stream_type, num_streams=2):
    """
    Build a collection of plots, callbacks, and streams for a given callback class and
    a list of trace_uids
    """
    plots = []
    streamss = []
    callbacks = []
    eventss = []
    for trace_uid in trace_uids:
        plot = mock_plot(trace_uid)
        streams, event_list = ([], [])
        for _ in range(num_streams):
            events = []
            stream = stream_type()

            def cb(events=events, **kwargs):
                events.append(kwargs)
            stream.add_subscriber(cb)
            streams.append(stream)
            event_list.append(events)
        callback = callback_cls(plot, streams, None)
        plots.append(plot)
        streamss.append(streams)
        callbacks.append(callback)
        eventss.append(event_list)
    return (plots, streamss, callbacks, eventss)