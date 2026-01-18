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
def info_header(label: str) -> str:
    """Make a nice header string."""
    return '--{:-<60s}'.format(' ' + label + ' ')