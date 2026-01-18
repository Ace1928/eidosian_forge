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
def set_accepted_encodings(encodings: typing.List[ContentEncoding]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets a list of content encodings that will be accepted. Empty list means no encoding is accepted.

    **EXPERIMENTAL**

    :param encodings: List of accepted content encodings.
    """
    params: T_JSON_DICT = dict()
    params['encodings'] = [i.to_json() for i in encodings]
    cmd_dict: T_JSON_DICT = {'method': 'Network.setAcceptedEncodings', 'params': params}
    json = (yield cmd_dict)