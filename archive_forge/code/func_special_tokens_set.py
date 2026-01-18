from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
@functools.cached_property
def special_tokens_set(self) -> set[str]:
    return set(self._special_tokens.keys())