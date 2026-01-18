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
def parseInline(self, src: str, env: EnvType | None=None) -> list[Token]:
    """The same as [[MarkdownIt.parse]] but skip all block rules.

        :param src: source string
        :param env: environment sandbox

        It returns the
        block tokens list with the single `inline` element, containing parsed inline
        tokens in `children` property. Also updates `env` object.
        """
    env = {} if env is None else env
    if not isinstance(env, MutableMapping):
        raise TypeError(f'Input data should be an MutableMapping, not {type(env)}')
    if not isinstance(src, str):
        raise TypeError(f'Input data should be a string, not {type(src)}')
    state = StateCore(src, self, env)
    state.inlineMode = True
    self.core.process(state)
    return state.tokens