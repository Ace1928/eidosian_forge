import contextlib
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import tempfile
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import (
import distutils.ccompiler
import distutils.errors
@lru_cache()
def auto_cpp_level(compiler: Any) -> Union[str, int]:
    """
    Return the max supported C++ std level (17, 14, or 11). Returns latest on Windows.
    """
    if WIN:
        return 'latest'
    levels = [17, 14, 11]
    for level in levels:
        if has_flag(compiler, STD_TMPL.format(level)):
            return level
    msg = 'Unsupported compiler -- at least C++11 support is needed!'
    raise RuntimeError(msg)