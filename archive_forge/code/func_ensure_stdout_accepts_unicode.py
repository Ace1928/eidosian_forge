from __future__ import annotations
from . import _pathlib
import sys
import os.path
import platform
import importlib
import argparse
import typing as T
from .utils.core import MesonException, MesonBugException
from . import mlog
def ensure_stdout_accepts_unicode() -> None:
    if sys.stdout.encoding and (not sys.stdout.encoding.upper().startswith('UTF-')):
        sys.stdout.reconfigure(errors='surrogateescape')