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
def set_attach_debug_stack(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Specifies whether to attach a page script stack id in requests

    **EXPERIMENTAL**

    :param enabled: Whether to attach a page script stack for debugging purpose.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Network.setAttachDebugStack', 'params': params}
    json = (yield cmd_dict)