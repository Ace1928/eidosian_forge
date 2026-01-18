from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_buildall_configurations_map(self) -> None:
    self.buildall_configurations = {self.buildtype: self.gen_id()}