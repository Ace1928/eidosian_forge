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
def run_runpython_command(self, options: argparse.Namespace) -> int:
    sys.argv[1:] = options.script_args
    if options.eval_arg:
        exec(options.script_file)
    else:
        import runpy
        sys.path.insert(0, os.path.dirname(options.script_file))
        runpy.run_path(options.script_file, run_name='__main__')
    return 0