from __future__ import annotations
import ctypes
import ctypes.util
import sys
import traceback
from functools import partial
from itertools import count
from threading import Lock, Thread
from typing import Any, Callable, Generic, TypeVar
import outcome
def darwin_namefunc(setname: Callable[[bytes], int], ident: int | None, name: str) -> None:
    if ident is not None:
        setname(_to_os_thread_name(name))