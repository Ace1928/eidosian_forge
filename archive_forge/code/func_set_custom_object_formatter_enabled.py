from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_custom_object_formatter_enabled(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """


    **EXPERIMENTAL**

    :param enabled:
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.setCustomObjectFormatterEnabled', 'params': params}
    json = (yield cmd_dict)