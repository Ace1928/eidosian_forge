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
def clear_accepted_encodings_override() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears accepted encodings set by setAcceptedEncodings

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Network.clearAcceptedEncodingsOverride'}
    json = (yield cmd_dict)