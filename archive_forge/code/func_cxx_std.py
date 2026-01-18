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
@cxx_std.setter
def cxx_std(self, level: int) -> None:
    if self._cxx_level:
        warnings.warn('You cannot safely change the cxx_level after setting it!', stacklevel=2)
    if WIN and level == 11:
        level = 14
    self._cxx_level = level
    if not level:
        return
    cflags = [STD_TMPL.format(level)]
    ldflags = []
    if MACOS and 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_macos = tuple((int(x) for x in platform.mac_ver()[0].split('.')[:2]))
        desired_macos = (10, 9) if level < 17 else (10, 14)
        macos_string = '.'.join((str(x) for x in min(current_macos, desired_macos)))
        macosx_min = f'-mmacosx-version-min={macos_string}'
        cflags += [macosx_min]
        ldflags += [macosx_min]
    self._add_cflags(cflags)
    self._add_ldflags(ldflags)