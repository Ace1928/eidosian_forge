from __future__ import annotations
import gc
import sys
from traceback import extract_tb
from typing import TYPE_CHECKING, Callable, NoReturn
import pytest
from .._concat_tb import concat_tb
def get_tb(raiser: Callable[[], NoReturn]) -> TracebackType | None:
    return get_exc(raiser).__traceback__