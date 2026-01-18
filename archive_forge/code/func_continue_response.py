from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
def continue_response(request_id: RequestId, response_code: typing.Optional[int]=None, response_phrase: typing.Optional[str]=None, response_headers: typing.Optional[typing.List[HeaderEntry]]=None, binary_response_headers: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Continues loading of the paused response, optionally modifying the
    response headers. If either responseCode or headers are modified, all of them
    must be present.

    **EXPERIMENTAL**

    :param request_id: An id the client received in requestPaused event.
    :param response_code: *(Optional)* An HTTP response code. If absent, original response code will be used.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    :param response_headers: *(Optional)* Response headers. If absent, original response headers will be used.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \x00-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    if response_code is not None:
        params['responseCode'] = response_code
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    cmd_dict: T_JSON_DICT = {'method': 'Fetch.continueResponse', 'params': params}
    json = (yield cmd_dict)