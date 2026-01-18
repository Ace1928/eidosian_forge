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
def can_clear_browser_cookies() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, bool]:
    """
    Tells whether clearing browser cookies is supported.

    :returns: True if browser cookies can be cleared.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Network.canClearBrowserCookies'}
    json = (yield cmd_dict)
    return bool(json['result'])