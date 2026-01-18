import os.path
import platform
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from pip._vendor.pygments.lexer import Lexer
from pip._vendor.pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pip._vendor.pygments.style import Style as PygmentsStyle
from pip._vendor.pygments.styles import get_style_by_name
from pip._vendor.pygments.token import (
from pip._vendor.pygments.util import ClassNotFound
from pip._vendor.rich.containers import Lines
from pip._vendor.rich.padding import Padding, PaddingDimensions
from ._loop import loop_first
from .cells import cell_len
from .color import Color, blend_rgb
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment, Segments
from .style import Style, StyleType
from .text import Text
def get_style_for_token(self, token_type: TokenType) -> Style:
    """Look up style in the style map."""
    try:
        return self._style_cache[token_type]
    except KeyError:
        get_style = self.style_map.get
        token = tuple(token_type)
        style = self._missing_style
        while token:
            _style = get_style(token)
            if _style is not None:
                style = _style
                break
            token = token[:-1]
        self._style_cache[token_type] = style
        return style