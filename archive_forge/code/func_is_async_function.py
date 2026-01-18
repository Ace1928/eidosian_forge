from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn
import py
def is_async_function(func: object) -> bool:
    """Return True if the given function seems to be an async function or
    an async generator."""
    return iscoroutinefunction(func) or inspect.isasyncgenfunction(func)