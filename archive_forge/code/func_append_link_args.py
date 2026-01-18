from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def append_link_args(self, args: T.List[str]) -> None:
    self.link_args += args