import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def match_library_callback(info, size, data):
    filepath = info.contents.dlpi_name
    if filepath:
        filepath = filepath.decode('utf-8')
        self._make_controller_from_path(filepath)
    return 0