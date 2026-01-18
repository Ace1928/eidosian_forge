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
def render_number(self, console: Console, options: ConsoleOptions, number: int, last_number: int) -> RenderResult:
    number_width = len(str(last_number)) + 2
    render_options = options.update(width=options.max_width - number_width)
    lines = console.render_lines(self.elements, render_options, style=self.style)
    number_style = console.get_style('markdown.item.number', default='none')
    new_line = Segment('\n')
    padding = Segment(' ' * number_width, number_style)
    numeral = Segment(f'{number}'.rjust(number_width - 1) + ' ', number_style)
    for first, line in loop_first(lines):
        yield (numeral if first else padding)
        yield from line
        yield new_line