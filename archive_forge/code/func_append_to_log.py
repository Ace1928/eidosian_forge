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
def append_to_log(lf: T.TextIO, line: str) -> None:
    lf.write(line)
    if not line.endswith('\n'):
        lf.write('\n')
    lf.flush()