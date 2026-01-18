from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_skip_all_pauses(skip: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Makes page not interrupt on any pauses (breakpoint, exception, dom exception etc).

    :param skip: New value for skip pauses state.
    """
    params: T_JSON_DICT = dict()
    params['skip'] = skip
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setSkipAllPauses', 'params': params}
    json = (yield cmd_dict)