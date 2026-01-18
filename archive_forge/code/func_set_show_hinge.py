from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_hinge(hinge_config: typing.Optional[HingeConfig]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Add a dual screen device hinge

    :param hinge_config: *(Optional)* hinge data, null means hideHinge
    """
    params: T_JSON_DICT = dict()
    if hinge_config is not None:
        params['hingeConfig'] = hinge_config.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowHinge', 'params': params}
    json = (yield cmd_dict)