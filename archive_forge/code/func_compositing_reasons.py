from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
def compositing_reasons(layer_id: LayerId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.List[str], typing.List[str]]]:
    """
    Provides the reasons why the given layer was composited.

    :param layer_id: The id of the layer for which we want to get the reasons it was composited.
    :returns: A tuple with the following items:

        0. **compositingReasons** - A list of strings specifying reasons for the given layer to become composited.
        1. **compositingReasonIds** - A list of strings specifying reason IDs for the given layer to become composited.
    """
    params: T_JSON_DICT = dict()
    params['layerId'] = layer_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'LayerTree.compositingReasons', 'params': params}
    json = (yield cmd_dict)
    return ([str(i) for i in json['compositingReasons']], [str(i) for i in json['compositingReasonIds']])