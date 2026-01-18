import inspect
import os
import platform
import sys
import threading
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from getpass import getpass
from html import escape
from inspect import isclass
from itertools import islice
from math import ceil
from time import monotonic
from types import FrameType, ModuleType, TracebackType
from typing import (
from pip._vendor.rich._null_file import NULL_FILE
from . import errors, themes
from ._emoji_replace import _emoji_replace
from ._export_format import CONSOLE_HTML_FORMAT, CONSOLE_SVG_FORMAT
from ._fileno import get_fileno
from ._log_render import FormatTimeCallable, LogRender
from .align import Align, AlignMethod
from .color import ColorSystem, blend_rgb
from .control import Control
from .emoji import EmojiVariant
from .highlighter import NullHighlighter, ReprHighlighter
from .markup import render as render_markup
from .measure import Measurement, measure_renderables
from .pager import Pager, SystemPager
from .pretty import Pretty, is_expandable
from .protocol import rich_cast
from .region import Region
from .scope import render_scope
from .screen import Screen
from .segment import Segment
from .style import Style, StyleType
from .styled import Styled
from .terminal_theme import DEFAULT_TERMINAL_THEME, SVG_EXPORT_THEME, TerminalTheme
from .text import Text, TextType
from .theme import Theme, ThemeStack
def render_str(self, text: str, *, style: Union[str, Style]='', justify: Optional[JustifyMethod]=None, overflow: Optional[OverflowMethod]=None, emoji: Optional[bool]=None, markup: Optional[bool]=None, highlight: Optional[bool]=None, highlighter: Optional[HighlighterType]=None) -> 'Text':
    """Convert a string to a Text instance. This is called automatically if
        you print or log a string.

        Args:
            text (str): Text to render.
            style (Union[str, Style], optional): Style to apply to rendered text.
            justify (str, optional): Justify method: "default", "left", "center", "full", or "right". Defaults to ``None``.
            overflow (str, optional): Overflow method: "crop", "fold", or "ellipsis". Defaults to ``None``.
            emoji (Optional[bool], optional): Enable emoji, or ``None`` to use Console default.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use Console default.
            highlight (Optional[bool], optional): Enable highlighting, or ``None`` to use Console default.
            highlighter (HighlighterType, optional): Optional highlighter to apply.
        Returns:
            ConsoleRenderable: Renderable object.

        """
    emoji_enabled = emoji or (emoji is None and self._emoji)
    markup_enabled = markup or (markup is None and self._markup)
    highlight_enabled = highlight or (highlight is None and self._highlight)
    if markup_enabled:
        rich_text = render_markup(text, style=style, emoji=emoji_enabled, emoji_variant=self._emoji_variant)
        rich_text.justify = justify
        rich_text.overflow = overflow
    else:
        rich_text = Text(_emoji_replace(text, default_variant=self._emoji_variant) if emoji_enabled else text, justify=justify, overflow=overflow, style=style)
    _highlighter = highlighter or self.highlighter if highlight_enabled else None
    if _highlighter is not None:
        highlight_text = _highlighter(str(rich_text))
        highlight_text.copy_styles(rich_text)
        return highlight_text
    return rich_text