from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_keyframe_key(style_sheet_id: StyleSheetId, range_: SourceRange, key_text: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, Value]:
    """
    Modifies the keyframe rule key text.

    :param style_sheet_id:
    :param range_:
    :param key_text:
    :returns: The resulting key text after modification.
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    params['range'] = range_.to_json()
    params['keyText'] = key_text
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setKeyframeKey', 'params': params}
    json = (yield cmd_dict)
    return Value.from_json(json['keyText'])