from __future__ import annotations
import dataclasses
import enum
import functools
import typing as T
from . import builder
from .. import mparser
from ..mesonlib import MesonBugException
@functools.singledispatch
def ir_to_meson(ir: T.Any, build: builder.Builder) -> mparser.BaseNode:
    raise NotImplementedError