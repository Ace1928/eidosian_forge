from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def get_manifest_icons() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[str]]:
    """
    Deprecated because it's not guaranteed that the returned icon is in fact the one used for PWA installation.

    **EXPERIMENTAL**

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getManifestIcons'}
    json = (yield cmd_dict)
    return str(json['primaryIcon']) if 'primaryIcon' in json else None