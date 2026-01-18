from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def terminate_execution() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Terminate current or next JavaScript execution.
    Will cancel the termination when the outer-most script execution ends.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.terminateExecution'}
    json = (yield cmd_dict)