import os.path
import platform
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pygments.style import Style as PygmentsStyle
from pygments.styles import get_style_by_name
from pygments.token import (
from pygments.util import ClassNotFound
from rich.containers import Lines
from rich.padding import Padding, PaddingDimensions
from ._loop import loop_first
from .cells import cell_len
from .color import Color, blend_rgb
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment, Segments
from .style import Style, StyleType
from .text import Text
@property
def default_lexer(self) -> Lexer:
    """A Pygments Lexer to use if one is not specified or invalid."""
    return get_lexer_by_name('text', stripnl=False, ensurenl=True, tabsize=self.tab_size)