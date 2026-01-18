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
def num_mock_patch_args(function) -> int:
    """Return number of arguments used up by mock arguments (if any)."""
    patchings = getattr(function, 'patchings', None)
    if not patchings:
        return 0
    mock_sentinel = getattr(sys.modules.get('mock'), 'DEFAULT', object())
    ut_mock_sentinel = getattr(sys.modules.get('unittest.mock'), 'DEFAULT', object())
    return len([p for p in patchings if not p.attribute_name and (p.new is mock_sentinel or p.new is ut_mock_sentinel)])