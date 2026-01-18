from asyncio import (
from collections import namedtuple
from collections.abc import Iterable
from functools import partial
from typing import List  # flake8: noqa
def iscoroutinefunctionorpartial(fn):
    return iscoroutinefunction(fn.func if isinstance(fn, partial) else fn)