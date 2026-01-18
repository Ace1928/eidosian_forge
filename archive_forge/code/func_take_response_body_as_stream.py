from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
def take_response_body_as_stream(request_id: RequestId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, io.StreamHandle]:
    """
    Returns a handle to the stream representing the response body.
    The request must be paused in the HeadersReceived stage.
    Note that after this command the request can't be continued
    as is -- client either needs to cancel it or to provide the
    response body.
    The stream only supports sequential read, IO.read will fail if the position
    is specified.
    This method is mutually exclusive with getResponseBody.
    Calling other methods that affect the request or disabling fetch
    domain before body is received results in an undefined behavior.

    :param request_id:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Fetch.takeResponseBodyAsStream', 'params': params}
    json = (yield cmd_dict)
    return io.StreamHandle.from_json(json['stream'])