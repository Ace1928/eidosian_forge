from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def discard_search_results(search_id: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Discards search results from the session with the given id. ``getSearchResults`` should no longer
    be called for that search.

    **EXPERIMENTAL**

    :param search_id: Unique search session identifier.
    """
    params: T_JSON_DICT = dict()
    params['searchId'] = search_id
    cmd_dict: T_JSON_DICT = {'method': 'DOM.discardSearchResults', 'params': params}
    json = (yield cmd_dict)