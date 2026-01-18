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
def set_ad_blocking_enabled(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enable Chrome's experimental ad filter on all sites.

    **EXPERIMENTAL**

    :param enabled: Whether to block ads.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Page.setAdBlockingEnabled', 'params': params}
    json = (yield cmd_dict)