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
def bytes_to_lines(code: CodeType) -> dict[int, int]:
    """Make a dict mapping byte code offsets to line numbers."""
    b2l = {}
    for bstart, bend, lineno in code.co_lines():
        if lineno is not None:
            for boffset in range(bstart, bend, 2):
                b2l[boffset] = lineno
    return b2l