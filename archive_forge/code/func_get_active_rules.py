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
def get_active_rules(self) -> dict[str, list[str]]:
    """Return the names of all active rules."""
    rules = {chain: self[chain].ruler.get_active_rules() for chain in ['core', 'block', 'inline']}
    rules['inline2'] = self.inline.ruler2.get_active_rules()
    return rules