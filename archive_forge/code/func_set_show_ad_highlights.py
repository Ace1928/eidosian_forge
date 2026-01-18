from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_ad_highlights(show: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Highlights owner element of all frames detected to be ads.

    :param show: True for showing ad highlights
    """
    params: T_JSON_DICT = dict()
    params['show'] = show
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowAdHighlights', 'params': params}
    json = (yield cmd_dict)