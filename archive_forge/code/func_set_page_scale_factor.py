from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_page_scale_factor(page_scale_factor: float) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets a specified page scale factor.

    **EXPERIMENTAL**

    :param page_scale_factor: Page scale factor.
    """
    params: T_JSON_DICT = dict()
    params['pageScaleFactor'] = page_scale_factor
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setPageScaleFactor', 'params': params}
    json = (yield cmd_dict)