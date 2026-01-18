from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, TypeVar
import json
from referencing import Resource
@cache
def cached_retrieve(uri: URI):
    response = retrieve(uri)
    contents = loads(response)
    return from_contents(contents)