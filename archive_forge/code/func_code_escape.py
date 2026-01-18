from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
def code_escape(text: str) -> str:
    """HTML escape a string of code."""
    if '&' in text:
        text = text.replace('&', '&amp;')
    if '<' in text:
        text = text.replace('<', '&lt;')
    if '>' in text:
        text = text.replace('>', '&gt;')
    return text