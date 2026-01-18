from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
def apply_skip(self, marked: str, reason: str, skipped: list[str], exclude: set[str]) -> None:
    """Apply the provided skips to the given exclude list."""
    if not skipped:
        return
    exclude.update(skipped)
    display.warning(f'Excluding {self.host_type} tests marked {marked} {reason}: {', '.join(skipped)}')