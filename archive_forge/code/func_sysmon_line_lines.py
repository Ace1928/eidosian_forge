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
@panopticon('code', 'line')
def sysmon_line_lines(self, code: CodeType, line_number: int) -> MonitorReturn:
    """Handle sys.monitoring.events.LINE events for line coverage."""
    code_info = self.code_infos[id(code)]
    if code_info.file_data is not None:
        cast(Set[TLineNo], code_info.file_data).add(line_number)
    return sys.monitoring.DISABLE