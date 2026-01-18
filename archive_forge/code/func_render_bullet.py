from __future__ import annotations
import sys
from typing import ClassVar, Dict, Iterable, List, Optional, Type, Union
from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.table import Table
from . import box
from ._loop import loop_first
from ._stack import Stack
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .containers import Renderables
from .jupyter import JupyterMixin
from .panel import Panel
from .rule import Rule
from .segment import Segment
from .style import Style, StyleStack
from .syntax import Syntax
from .text import Text, TextType
def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
    render_options = options.update(width=options.max_width - 3)
    lines = console.render_lines(self.elements, render_options, style=self.style)
    bullet_style = console.get_style('markdown.item.bullet', default='none')
    bullet = Segment(' • ', bullet_style)
    padding = Segment(' ' * 3, bullet_style)
    new_line = Segment('\n')
    for first, line in loop_first(lines):
        yield (bullet if first else padding)
        yield from line
        yield new_line