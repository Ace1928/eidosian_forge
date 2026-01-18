from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
def get_crt_val(self, crt_val: str, buildtype: str) -> str:
    if crt_val in MSCRT_VALS:
        return crt_val
    assert crt_val in {'from_buildtype', 'static_from_buildtype'}
    dbg = 'mdd'
    rel = 'md'
    if crt_val == 'static_from_buildtype':
        dbg = 'mtd'
        rel = 'mt'
    if buildtype == 'plain':
        return 'none'
    elif buildtype == 'debug':
        return dbg
    elif buildtype in {'debugoptimized', 'release', 'minsize'}:
        return rel
    else:
        assert buildtype == 'custom'
        raise EnvironmentException('Requested C runtime based on buildtype, but buildtype is "custom".')