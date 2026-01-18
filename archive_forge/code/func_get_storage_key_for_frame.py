from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def get_storage_key_for_frame(frame_id: page.FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SerializedStorageKey]:
    """
    Returns a storage key given a frame id.

    :param frame_id:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Storage.getStorageKeyForFrame', 'params': params}
    json = (yield cmd_dict)
    return SerializedStorageKey.from_json(json['storageKey'])