from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_emulated_vision_deficiency(type_: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Emulates the given vision deficiency.

    :param type_: Vision deficiency to emulate. Order: best-effort emulations come first, followed by any physiologically accurate emulations for medically recognized color vision deficiencies.
    """
    params: T_JSON_DICT = dict()
    params['type'] = type_
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setEmulatedVisionDeficiency', 'params': params}
    json = (yield cmd_dict)