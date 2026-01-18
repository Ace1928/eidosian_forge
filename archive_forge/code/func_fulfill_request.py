from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
def fulfill_request(request_id: RequestId, response_code: int, response_headers: typing.Optional[typing.List[HeaderEntry]]=None, binary_response_headers: typing.Optional[str]=None, body: typing.Optional[str]=None, response_phrase: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Provides response to the request.

    :param request_id: An id the client received in requestPaused event.
    :param response_code: An HTTP response code.
    :param response_headers: *(Optional)* Response headers.
    :param binary_response_headers: *(Optional)* Alternative way of specifying response headers as a \x00-separated series of name: value pairs. Prefer the above method unless you need to represent some non-UTF8 values that can't be transmitted over the protocol as text.
    :param body: *(Optional)* A response body.
    :param response_phrase: *(Optional)* A textual representation of responseCode. If absent, a standard phrase matching responseCode is used.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['responseCode'] = response_code
    if response_headers is not None:
        params['responseHeaders'] = [i.to_json() for i in response_headers]
    if binary_response_headers is not None:
        params['binaryResponseHeaders'] = binary_response_headers
    if body is not None:
        params['body'] = body
    if response_phrase is not None:
        params['responsePhrase'] = response_phrase
    cmd_dict: T_JSON_DICT = {'method': 'Fetch.fulfillRequest', 'params': params}
    json = (yield cmd_dict)