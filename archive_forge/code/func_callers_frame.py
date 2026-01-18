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
def callers_frame(self) -> FrameType:
    """Get the frame of the Python code we're monitoring."""
    return inspect.currentframe().f_back.f_back