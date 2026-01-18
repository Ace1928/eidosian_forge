from __future__ import annotations
import asyncio
import functools
import re
import sys
import typing
from contextlib import contextmanager
from starlette.types import Scope
@contextmanager
def collapse_excgroups() -> typing.Generator[None, None, None]:
    try:
        yield
    except BaseException as exc:
        if has_exceptiongroups:
            while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]
        raise exc