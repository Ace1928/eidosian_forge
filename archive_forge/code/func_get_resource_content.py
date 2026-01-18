from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def get_resource_content(frame_id: FrameId, url: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, bool]]:
    """
    Returns content of the given resource.

    **EXPERIMENTAL**

    :param frame_id: Frame id to get resource for.
    :param url: URL of the resource to get content for.
    :returns: A tuple with the following items:

        0. **content** - Resource content.
        1. **base64Encoded** - True, if content was served as base64.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {'method': 'Page.getResourceContent', 'params': params}
    json = (yield cmd_dict)
    return (str(json['content']), bool(json['base64Encoded']))