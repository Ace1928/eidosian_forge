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
def search_in_response_body(request_id: RequestId, query: str, case_sensitive: typing.Optional[bool]=None, is_regex: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[debugger.SearchMatch]]:
    """
    Searches for given string in response content.

    **EXPERIMENTAL**

    :param request_id: Identifier of the network response to search.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {'method': 'Network.searchInResponseBody', 'params': params}
    json = (yield cmd_dict)
    return [debugger.SearchMatch.from_json(i) for i in json['result']]