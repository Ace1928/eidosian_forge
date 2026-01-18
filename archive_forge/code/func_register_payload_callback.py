from __future__ import annotations
import collections.abc as c
import dataclasses
import os
import typing as t
from .util import (
from .provider import (
from .provider.source import (
from .provider.source.unversioned import (
from .provider.source.installed import (
from .provider.source.unsupported import (
from .provider.layout import (
from .provider.layout.unsupported import (
def register_payload_callback(self, callback: c.Callable[[PayloadConfig], None]) -> None:
    """Register the given payload callback."""
    self.payload_callbacks.append(callback)