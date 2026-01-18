from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_native_target_map(self) -> None:
    self.native_targets = {}
    for t in self.build_targets:
        self.native_targets[t] = self.gen_id()