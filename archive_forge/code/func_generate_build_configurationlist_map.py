from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_build_configurationlist_map(self) -> None:
    self.buildconflistmap = {}
    for t in self.build_targets:
        self.buildconflistmap[t] = self.gen_id()
    for t in self.custom_targets:
        self.buildconflistmap[t] = self.gen_id()