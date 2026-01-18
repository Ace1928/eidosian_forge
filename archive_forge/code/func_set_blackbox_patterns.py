from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_blackbox_patterns(patterns: typing.List[str]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Replace previous blackbox patterns with passed ones. Forces backend to skip stepping/pausing in
    scripts with url matching one of the patterns. VM will try to leave blackboxed script by
    performing 'step in' several times, finally resorting to 'step out' if unsuccessful.

    **EXPERIMENTAL**

    :param patterns: Array of regexps that will be used to check script url for blackbox state.
    """
    params: T_JSON_DICT = dict()
    params['patterns'] = [i for i in patterns]
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setBlackboxPatterns', 'params': params}
    json = (yield cmd_dict)