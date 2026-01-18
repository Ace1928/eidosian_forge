from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def create_list_rule(src: str, pos: Pos, out: Output) -> tuple[Pos, Key]:
    pos += 2
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)
    if out.flags.is_(key, Flags.FROZEN):
        raise suffixed_err(src, pos, f'Cannot mutate immutable namespace {key}')
    out.flags.unset_all(key)
    out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        out.data.append_nest_to_list(key)
    except KeyError:
        raise suffixed_err(src, pos, 'Cannot overwrite a value') from None
    if not src.startswith(']]', pos):
        raise suffixed_err(src, pos, "Expected ']]' at the end of an array declaration")
    return (pos + 2, key)