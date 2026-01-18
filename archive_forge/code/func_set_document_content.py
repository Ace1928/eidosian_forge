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
def set_document_content(frame_id: FrameId, html: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets given markup as the document's HTML.

    :param frame_id: Frame id to set HTML for.
    :param html: HTML content to set.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    params['html'] = html
    cmd_dict: T_JSON_DICT = {'method': 'Page.setDocumentContent', 'params': params}
    json = (yield cmd_dict)