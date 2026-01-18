from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
def get_manifest_for_frame(frame_id: page.FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Returns manifest URL for document in the given frame.

    :param frame_id: Identifier of the frame containing document whose manifest is retrieved.
    :returns: Manifest URL for document in the given frame.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'ApplicationCache.getManifestForFrame', 'params': params}
    json = (yield cmd_dict)
    return str(json['manifestURL'])