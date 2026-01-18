from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_pbxdep_map(self) -> None:
    self.pbx_dep_map = {}
    self.pbx_custom_dep_map = {}
    for t in self.build_targets:
        self.pbx_dep_map[t] = self.gen_id()
    for t in self.custom_targets:
        self.pbx_custom_dep_map[t] = self.gen_id()