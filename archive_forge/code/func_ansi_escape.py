from __future__ import annotations
from string import Formatter
from typing import Generator
from prompt_toolkit.output.vt100 import BG_ANSI_COLORS, FG_ANSI_COLORS
from prompt_toolkit.output.vt100 import _256_colors as _256_colors_table
from .base import StyleAndTextTuples
def ansi_escape(text: object) -> str:
    """
    Replace characters with a special meaning.
    """
    return str(text).replace('\x1b', '?').replace('\x08', '?')