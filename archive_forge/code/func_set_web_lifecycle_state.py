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
def set_web_lifecycle_state(state: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Tries to update the web lifecycle state of the page.
    It will transition the page to the given state according to:
    https://github.com/WICG/web-lifecycle/

    **EXPERIMENTAL**

    :param state: Target lifecycle state
    """
    params: T_JSON_DICT = dict()
    params['state'] = state
    cmd_dict: T_JSON_DICT = {'method': 'Page.setWebLifecycleState', 'params': params}
    json = (yield cmd_dict)