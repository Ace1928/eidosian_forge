from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_media_queries() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[CSSMedia]]:
    """
    Returns all media queries parsed by the rendering engine.

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getMediaQueries'}
    json = (yield cmd_dict)
    return [CSSMedia.from_json(i) for i in json['medias']]