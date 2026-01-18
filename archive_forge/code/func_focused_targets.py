from __future__ import annotations
import typing as t
from .util import (
from .io import (
from .diff import (
@property
def focused_targets(self) -> t.Optional[list[str]]:
    """Optional list of focused target names."""
    return self.focused_command_targets.get(self.command)