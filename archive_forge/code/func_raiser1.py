from __future__ import annotations
import gc
import sys
from traceback import extract_tb
from typing import TYPE_CHECKING, Callable, NoReturn
import pytest
from .._concat_tb import concat_tb
def raiser1() -> NoReturn:
    raiser1_2()