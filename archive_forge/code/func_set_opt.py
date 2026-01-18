from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def set_opt(self, opt: str, val: str) -> None:
    self.opts[opt] = val