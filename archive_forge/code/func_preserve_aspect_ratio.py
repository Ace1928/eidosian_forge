from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def preserve_aspect_ratio():

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)
    x, y = provided_size
    if x >= self.width and y >= self.height:
        return
    aspect = self.width / self.height
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    return (x, y)