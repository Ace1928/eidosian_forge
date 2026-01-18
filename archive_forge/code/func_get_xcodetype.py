from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def get_xcodetype(self, fname: str) -> str:
    extension = fname.split('.')[-1]
    if extension == 'C':
        extension = 'cpp'
    xcodetype = XCODETYPEMAP.get(extension.lower())
    if not xcodetype:
        xcodetype = 'sourcecode.unknown'
    return xcodetype