from __future__ import annotations
import functools
import inspect
import os
import os.path
import sys
import threading
import traceback
from dataclasses import dataclass
from types import CodeType, FrameType
from typing import (
from coverage.debug import short_filename, short_stack
from coverage.types import (
@panopticon('code', '@', 'exc')
def sysmon_py_unwind_arcs(self, code: CodeType, instruction_offset: int, exception: BaseException) -> MonitorReturn:
    """Handle sys.monitoring.events.PY_UNWIND events for branch coverage."""
    frame = self.callers_frame()
    last_line = self.last_lines.pop(frame, None)
    if isinstance(exception, GeneratorExit):
        return
    code_info = self.code_infos.get(id(code))
    if code_info is not None and code_info.file_data is not None:
        if last_line is not None:
            arc = (last_line, -code.co_firstlineno)
            cast(Set[TArc], code_info.file_data).add(arc)