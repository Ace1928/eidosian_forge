from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
def noSecondLevelHolderResolving(f: TV_func) -> TV_func:
    setattr(f, 'no-second-level-holder-flattening', True)
    return f