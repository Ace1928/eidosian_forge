from __future__ import annotations
import os
import os.path
import sys
from types import FrameType
from typing import Any, Iterable, Iterator
from coverage.exceptions import PluginError
from coverage.misc import isolate_module
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter
from coverage.types import (
def dynamic_context(self, frame: FrameType) -> str | None:
    context = self.plugin.dynamic_context(frame)
    self.debug.write(f'dynamic_context({frame!r}) --> {context!r}')
    return context