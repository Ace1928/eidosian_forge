from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_scrollbars_hidden(hidden: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """


    **EXPERIMENTAL**

    :param hidden: Whether scrollbars should be always hidden.
    """
    params: T_JSON_DICT = dict()
    params['hidden'] = hidden
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setScrollbarsHidden', 'params': params}
    json = (yield cmd_dict)