from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
@classmethod
def get_event_data_from_property_update(cls, property, property_value, fig_dict):
    traces = fig_dict.get('data', [])
    if property == 'viewport':
        event_data = cls.build_event_data_from_viewport(traces, property_value)
    else:
        event_data = cls.build_event_data_from_relayout_data(traces, property_value)
    return event_data