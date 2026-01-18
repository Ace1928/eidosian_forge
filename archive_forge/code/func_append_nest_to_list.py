from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def append_nest_to_list(self, key: Key) -> None:
    cont = self.get_or_create_nest(key[:-1])
    last_key = key[-1]
    if last_key in cont:
        list_ = cont[last_key]
        if not isinstance(list_, list):
            raise KeyError('An object other than list found behind this key')
        list_.append({})
    else:
        cont[last_key] = [{}]