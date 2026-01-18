from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_realtime_data(context_id: GraphObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, ContextRealtimeData]:
    """
    Fetch the realtime data from the registered contexts.

    :param context_id:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['contextId'] = context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'WebAudio.getRealtimeData', 'params': params}
    json = (yield cmd_dict)
    return ContextRealtimeData.from_json(json['realtimeData'])