from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def create_style_sheet(frame_id: page.FrameId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, StyleSheetId]:
    """
    Creates a new special "via-inspector" stylesheet in the frame with given ``frameId``.

    :param frame_id: Identifier of the frame where "via-inspector" stylesheet should be created.
    :returns: Identifier of the created "via-inspector" stylesheet.
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.createStyleSheet', 'params': params}
    json = (yield cmd_dict)
    return StyleSheetId.from_json(json['styleSheetId'])