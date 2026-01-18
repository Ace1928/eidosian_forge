from __future__ import annotations
from glob import glob
import argparse
import errno
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import typing as T
import re
from . import build, environment
from .backend.backends import InstallData
from .mesonlib import (MesonException, Popen_safe, RealPathAction, is_windows,
from .scripts import depfixer, destdir_join
from .scripts.meson_exe import run_exe
def should_preserve_existing_file(self, from_file: str, to_file: str) -> bool:
    if not self.options.only_changed:
        return False
    if os.path.islink(from_file) and (not os.path.isfile(from_file)):
        return False
    from_time = os.stat(from_file).st_mtime
    to_time = os.stat(to_file).st_mtime
    return from_time <= to_time