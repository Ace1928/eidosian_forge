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
def stream_resource_content(request_id: RequestId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Enables streaming of the response for the given requestId.
    If enabled, the dataReceived event contains the data that was received during streaming.

    **EXPERIMENTAL**

    :param request_id: Identifier of the request to stream.
    :returns: Data that has been buffered until streaming is enabled.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.streamResourceContent', 'params': params}
    json = (yield cmd_dict)
    return str(json['bufferedData'])