from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
@classmethod
def build_event_data_from_viewport(cls, traces, property_value):
    event_data = {}
    for trace in traces:
        trace_type = trace.get('type', 'scatter')
        trace_uid = trace.get('uid', None)
        if _trace_to_subplot.get(trace_type, None) != ['xaxis', 'yaxis']:
            continue
        xaxis = trace.get('xaxis', 'x').replace('x', 'xaxis')
        yaxis = trace.get('yaxis', 'y').replace('y', 'yaxis')
        xprop = f'{xaxis}.range'
        yprop = f'{yaxis}.range'
        if not property_value:
            x_range = None
            y_range = None
        elif xprop in property_value and yprop in property_value:
            x_range = tuple(property_value[xprop])
            y_range = tuple(property_value[yprop])
        elif xprop + '[0]' in property_value and xprop + '[1]' in property_value and (yprop + '[0]' in property_value) and (yprop + '[1]' in property_value):
            x_range = (property_value[xprop + '[0]'], property_value[xprop + '[1]'])
            y_range = (property_value[yprop + '[0]'], property_value[yprop + '[1]'])
        else:
            continue
        stream_data = {}
        if cls.x_range:
            stream_data['x_range'] = x_range
        if cls.y_range:
            stream_data['y_range'] = y_range
        event_data[trace_uid] = stream_data
    return event_data