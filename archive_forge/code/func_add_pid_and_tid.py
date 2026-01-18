from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
def add_pid_and_tid(text: str) -> str:
    """A filter to add pid and tid to debug messages."""
    tid = f'{short_id(_thread.get_ident()):04x}'
    text = f'{os.getpid():5d}.{tid}: {text}'
    return text