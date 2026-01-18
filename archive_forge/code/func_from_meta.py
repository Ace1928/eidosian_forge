import sys
from functools import lru_cache
from marshal import dumps, loads
from random import randint
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast
from . import errors
from .color import Color, ColorParseError, ColorSystem, blend_rgb
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME, TerminalTheme
@classmethod
def from_meta(cls, meta: Optional[Dict[str, Any]]) -> 'Style':
    """Create a new style with meta data.

        Returns:
            meta (Optional[Dict[str, Any]]): A dictionary of meta data. Defaults to None.
        """
    style: Style = cls.__new__(Style)
    style._ansi = None
    style._style_definition = None
    style._color = None
    style._bgcolor = None
    style._set_attributes = 0
    style._attributes = 0
    style._link = None
    style._meta = dumps(meta)
    style._link_id = f'{randint(0, 999999)}{hash(style._meta)}'
    style._hash = None
    style._null = not meta
    return style