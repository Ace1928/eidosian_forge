from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_hardware_concurrency_override(hardware_concurrency: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """


    **EXPERIMENTAL**

    :param hardware_concurrency: Hardware concurrency to report
    """
    params: T_JSON_DICT = dict()
    params['hardwareConcurrency'] = hardware_concurrency
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setHardwareConcurrencyOverride', 'params': params}
    json = (yield cmd_dict)