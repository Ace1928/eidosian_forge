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
def set_rph_registration_mode(mode: AutoResponseMode) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Extensions for Custom Handlers API:
    https://html.spec.whatwg.org/multipage/system-state.html#rph-automation

    **EXPERIMENTAL**

    :param mode:
    """
    params: T_JSON_DICT = dict()
    params['mode'] = mode.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.setRPHRegistrationMode', 'params': params}
    json = (yield cmd_dict)