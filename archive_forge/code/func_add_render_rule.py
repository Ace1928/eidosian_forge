from __future__ import annotations
from collections.abc import Callable, Generator, Iterable, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any, Literal, overload
from . import helpers, presets
from .common import normalize_url, utils
from .parser_block import ParserBlock
from .parser_core import ParserCore
from .parser_inline import ParserInline
from .renderer import RendererHTML, RendererProtocol
from .rules_core.state_core import StateCore
from .token import Token
from .utils import EnvType, OptionsDict, OptionsType, PresetType
def add_render_rule(self, name: str, function: Callable[..., Any], fmt: str='html') -> None:
    """Add a rule for rendering a particular Token type.

        Only applied when ``renderer.__output__ == fmt``
        """
    if self.renderer.__output__ == fmt:
        self.renderer.rules[name] = function.__get__(self.renderer)