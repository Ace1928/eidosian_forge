from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_layout_shift_regions(result: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Requests that backend shows layout shift regions

    :param result: True for showing layout shift regions
    """
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowLayoutShiftRegions', 'params': params}
    json = (yield cmd_dict)