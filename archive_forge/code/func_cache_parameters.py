import asyncio
import dataclasses
import sys
from asyncio.coroutines import _is_coroutine  # type: ignore[attr-defined]
from functools import _CacheInfo, _make_key, partial, partialmethod
from typing import (
def cache_parameters(self) -> _CacheParameters:
    return self.__wrapper.cache_parameters()