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
def set_extra_http_headers(headers: Headers) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Specifies whether to always send extra HTTP headers with the requests from this page.

    :param headers: Map with extra HTTP headers.
    """
    params: T_JSON_DICT = dict()
    params['headers'] = headers.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.setExtraHTTPHeaders', 'params': params}
    json = (yield cmd_dict)