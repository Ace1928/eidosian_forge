from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def search_in_content(script_id: runtime.ScriptId, query: str, case_sensitive: typing.Optional[bool]=None, is_regex: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[SearchMatch]]:
    """
    Searches for given string in script content.

    :param script_id: Id of the script to search in.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    """
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.searchInContent', 'params': params}
    json = (yield cmd_dict)
    return [SearchMatch.from_json(i) for i in json['result']]