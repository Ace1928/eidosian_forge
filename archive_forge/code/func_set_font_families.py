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
def set_font_families(font_families: FontFamilies, for_scripts: typing.Optional[typing.List[ScriptFontFamilies]]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Set generic font families.

    **EXPERIMENTAL**

    :param font_families: Specifies font families to set. If a font family is not specified, it won't be changed.
    :param for_scripts: *(Optional)* Specifies font families to set for individual scripts.
    """
    params: T_JSON_DICT = dict()
    params['fontFamilies'] = font_families.to_json()
    if for_scripts is not None:
        params['forScripts'] = [i.to_json() for i in for_scripts]
    cmd_dict: T_JSON_DICT = {'method': 'Page.setFontFamilies', 'params': params}
    json = (yield cmd_dict)