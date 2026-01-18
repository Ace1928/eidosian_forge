from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import runtime
def start_violations_report(config: typing.List[ViolationSetting]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    start violation reporting.

    :param config: Configuration for violations.
    """
    params: T_JSON_DICT = dict()
    params['config'] = [i.to_json() for i in config]
    cmd_dict: T_JSON_DICT = {'method': 'Log.startViolationsReport', 'params': params}
    json = (yield cmd_dict)