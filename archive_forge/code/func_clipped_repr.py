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
def clipped_repr(text: str, numchars: int=50) -> str:
    """`repr(text)`, but limited to `numchars`."""
    r = reprlib.Repr()
    r.maxstring = numchars
    return r.repr(text)