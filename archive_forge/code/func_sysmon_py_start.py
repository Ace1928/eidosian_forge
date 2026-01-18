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
@panopticon('code', '@')
def sysmon_py_start(self, code: CodeType, instruction_offset: int) -> MonitorReturn:
    """Handle sys.monitoring.events.PY_START events."""
    self._activity = True
    self.stats['starts'] += 1
    code_info = self.code_infos.get(id(code))
    tracing_code: bool | None = None
    file_data: TTraceFileData | None = None
    if code_info is not None:
        tracing_code = code_info.tracing
        file_data = code_info.file_data
    if tracing_code is None:
        filename = code.co_filename
        disp = self.should_trace_cache.get(filename)
        if disp is None:
            frame = inspect.currentframe().f_back
            if LOG:
                frame = frame.f_back
            disp = self.should_trace(filename, frame)
            self.should_trace_cache[filename] = disp
        tracing_code = disp.trace
        if tracing_code:
            tracename = disp.source_filename
            assert tracename is not None
            if tracename not in self.data:
                self.data[tracename] = set()
            file_data = self.data[tracename]
            b2l = bytes_to_lines(code)
        else:
            file_data = None
            b2l = None
        self.code_infos[id(code)] = CodeInfo(tracing=tracing_code, file_data=file_data, byte_to_line=b2l)
        self.code_objects.append(code)
        if tracing_code:
            events = sys.monitoring.events
            if self.sysmon_on:
                assert sys_monitoring is not None
                sys_monitoring.set_local_events(self.myid, code, events.PY_RETURN | events.PY_RESUME | events.LINE)
                self.local_event_codes[id(code)] = code
    if tracing_code and self.trace_arcs:
        frame = self.callers_frame()
        self.last_lines[frame] = -code.co_firstlineno
        return None
    else:
        return sys.monitoring.DISABLE