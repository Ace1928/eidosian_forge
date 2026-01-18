from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
def get_application_cache_for_frame(frame_id: page.FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, ApplicationCache]:
    """
    Returns relevant application cache data for the document in given frame.

    :param frame_id: Identifier of the frame containing document whose application cache is retrieved.
    :returns: Relevant application cache data for the document in given frame.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'ApplicationCache.getApplicationCacheForFrame', 'params': params}
    json = (yield cmd_dict)
    return ApplicationCache.from_json(json['applicationCache'])