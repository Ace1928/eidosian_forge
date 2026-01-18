from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import runtime
def stop_violations_report() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Stop violation reporting.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Log.stopViolationsReport'}
    json = (yield cmd_dict)