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
def screencast_frame_ack(session_id: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Acknowledges that a screencast frame has been received by the frontend.

    **EXPERIMENTAL**

    :param session_id: Frame number.
    """
    params: T_JSON_DICT = dict()
    params['sessionId'] = session_id
    cmd_dict: T_JSON_DICT = {'method': 'Page.screencastFrameAck', 'params': params}
    json = (yield cmd_dict)