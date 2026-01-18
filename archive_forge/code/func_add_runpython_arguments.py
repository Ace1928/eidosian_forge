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
def add_runpython_arguments(self, parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-c', action='store_true', dest='eval_arg', default=False)
    parser.add_argument('--version', action='version', version=platform.python_version())
    parser.add_argument('script_file')
    parser.add_argument('script_args', nargs=argparse.REMAINDER)