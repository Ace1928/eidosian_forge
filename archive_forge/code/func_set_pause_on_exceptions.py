from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def set_pause_on_exceptions(state: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Defines pause on exceptions state. Can be set to stop on all exceptions, uncaught exceptions or
    no exceptions. Initial pause on exceptions state is ``none``.

    :param state: Pause on exceptions mode.
    """
    params: T_JSON_DICT = dict()
    params['state'] = state
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.setPauseOnExceptions', 'params': params}
    json = (yield cmd_dict)