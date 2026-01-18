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
def set_lifecycle_events_enabled(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Controls whether page will emit lifecycle events.

    :param enabled: If true, starts emitting lifecycle events.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Page.setLifecycleEventsEnabled', 'params': params}
    json = (yield cmd_dict)