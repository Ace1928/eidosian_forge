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
def line_number_range(self, frame: FrameType) -> tuple[TLineNo, TLineNo]:
    pair = self.tracer.line_number_range(frame)
    self.debug.write(f'line_number_range({self._show_frame(frame)}) --> {pair!r}')
    return pair