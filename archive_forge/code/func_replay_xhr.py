from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def replay_xhr(request_id: RequestId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    This method sends a new XMLHttpRequest which is identical to the original one. The following
    parameters should be identical: method, url, async, request body, extra headers, withCredentials
    attribute, user, password.

    **EXPERIMENTAL**

    :param request_id: Identifier of XHR to replay.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.replayXHR', 'params': params}
    json = (yield cmd_dict)