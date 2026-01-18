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
def run_help_command(self, options: argparse.Namespace) -> int:
    if options.command:
        self.commands[options.command].print_help()
    else:
        self.parser.print_help()
    return 0