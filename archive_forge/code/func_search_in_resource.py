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
def search_in_resource(frame_id: FrameId, url: str, query: str, case_sensitive: typing.Optional[bool]=None, is_regex: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[debugger.SearchMatch]]:
    """
    Searches for given string in resource content.

    **EXPERIMENTAL**

    :param frame_id: Frame id for resource to search in.
    :param url: URL of the resource to search in.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    params['url'] = url
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {'method': 'Page.searchInResource', 'params': params}
    json = (yield cmd_dict)
    return [debugger.SearchMatch.from_json(i) for i in json['result']]