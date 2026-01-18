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
def legacy_path(path: str | os.PathLike[str]) -> LEGACY_PATH:
    """Internal wrapper to prepare lazy proxies for legacy_path instances"""
    return LEGACY_PATH(path)