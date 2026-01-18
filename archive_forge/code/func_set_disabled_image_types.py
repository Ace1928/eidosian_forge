from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_disabled_image_types(image_types: typing.List[DisabledImageType]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """


    **EXPERIMENTAL**

    :param image_types: Image types to disable.
    """
    params: T_JSON_DICT = dict()
    params['imageTypes'] = [i.to_json() for i in image_types]
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setDisabledImageTypes', 'params': params}
    json = (yield cmd_dict)