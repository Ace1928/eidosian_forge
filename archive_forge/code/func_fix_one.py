from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
def fix_one(qualname: str, name: str, obj: object) -> None:
    if id(obj) in seen_ids:
        return
    seen_ids.add(id(obj))
    mod = getattr(obj, '__module__', None)
    if mod is not None and mod.startswith('trio.'):
        obj.__module__ = module_name
        if hasattr(obj, '__name__') and '.' not in obj.__name__:
            obj.__name__ = name
            if hasattr(obj, '__qualname__'):
                obj.__qualname__ = qualname
        if isinstance(obj, type):
            for attr_name, attr_value in obj.__dict__.items():
                fix_one(objname + '.' + attr_name, attr_name, attr_value)