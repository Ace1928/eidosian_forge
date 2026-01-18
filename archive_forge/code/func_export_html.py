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
def export_html(self, *, theme: Optional[TerminalTheme]=None, clear: bool=True, code_format: Optional[str]=None, inline_styles: bool=False) -> str:
    """Generate HTML from console contents (requires record=True argument in constructor).

        Args:
            theme (TerminalTheme, optional): TerminalTheme object containing console colors.
            clear (bool, optional): Clear record buffer after exporting. Defaults to ``True``.
            code_format (str, optional): Format string to render HTML. In addition to '{foreground}',
                '{background}', and '{code}', should contain '{stylesheet}' if inline_styles is ``False``.
            inline_styles (bool, optional): If ``True`` styles will be inlined in to spans, which makes files
                larger but easier to cut and paste markup. If ``False``, styles will be embedded in a style tag.
                Defaults to False.

        Returns:
            str: String containing console contents as HTML.
        """
    assert self.record, 'To export console contents set record=True in the constructor or instance'
    fragments: List[str] = []
    append = fragments.append
    _theme = theme or DEFAULT_TERMINAL_THEME
    stylesheet = ''
    render_code_format = CONSOLE_HTML_FORMAT if code_format is None else code_format
    with self._record_buffer_lock:
        if inline_styles:
            for text, style, _ in Segment.filter_control(Segment.simplify(self._record_buffer)):
                text = escape(text)
                if style:
                    rule = style.get_html_style(_theme)
                    if style.link:
                        text = f'<a href="{style.link}">{text}</a>'
                    text = f'<span style="{rule}">{text}</span>' if rule else text
                append(text)
        else:
            styles: Dict[str, int] = {}
            for text, style, _ in Segment.filter_control(Segment.simplify(self._record_buffer)):
                text = escape(text)
                if style:
                    rule = style.get_html_style(_theme)
                    style_number = styles.setdefault(rule, len(styles) + 1)
                    if style.link:
                        text = f'<a class="r{style_number}" href="{style.link}">{text}</a>'
                    else:
                        text = f'<span class="r{style_number}">{text}</span>'
                append(text)
            stylesheet_rules: List[str] = []
            stylesheet_append = stylesheet_rules.append
            for style_rule, style_number in styles.items():
                if style_rule:
                    stylesheet_append(f'.r{style_number} {{{style_rule}}}')
            stylesheet = '\n'.join(stylesheet_rules)
        rendered_code = render_code_format.format(code=''.join(fragments), stylesheet=stylesheet, foreground=_theme.foreground_color.hex, background=_theme.background_color.hex)
        if clear:
            del self._record_buffer[:]
    return rendered_code