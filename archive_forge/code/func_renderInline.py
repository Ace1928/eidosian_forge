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
def renderInline(self, src: str, env: EnvType | None=None) -> Any:
    """Similar to [[MarkdownIt.render]] but for single paragraph content.

        :param src: source string
        :param env: environment sandbox

        Similar to [[MarkdownIt.render]] but for single paragraph content. Result
        will NOT be wrapped into `<p>` tags.
        """
    env = {} if env is None else env
    return self.renderer.render(self.parseInline(src, env), self.options, env)