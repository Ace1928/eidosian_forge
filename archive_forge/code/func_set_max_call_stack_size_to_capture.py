from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_max_call_stack_size_to_capture(size: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """


    **EXPERIMENTAL**

    :param size:
    """
    params: T_JSON_DICT = dict()
    params['size'] = size
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.setMaxCallStackSizeToCapture', 'params': params}
    json = (yield cmd_dict)