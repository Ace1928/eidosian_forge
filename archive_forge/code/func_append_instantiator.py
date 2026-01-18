from __future__ import annotations
import argparse
import dataclasses
import enum
import functools
import itertools
import json
import shlex
from typing import (
import rich.markup
import shtab
from . import _fields, _instantiators, _resolver, _strings
from ._typing import TypeForm
from .conf import _markers
def append_instantiator(x: Any) -> Any:
    out = instantiator(x)
    if arg.field.default in _fields.MISSING_SINGLETONS:
        return instantiator(x)
    return type(out)(arg.field.default) + out