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
def clear_browser_cache() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears browser cache.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Network.clearBrowserCache'}
    json = (yield cmd_dict)