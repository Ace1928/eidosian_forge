import sys
from functools import lru_cache
from marshal import dumps, loads
from random import randint
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast
from . import errors
from .color import Color, ColorParseError, ColorSystem, blend_rgb
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME, TerminalTheme
@property
def background_style(self) -> 'Style':
    """A Style with background only."""
    return Style(bgcolor=self.bgcolor)