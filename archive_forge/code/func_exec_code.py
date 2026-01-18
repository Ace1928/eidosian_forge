import argparse
from typing import Tuple, List, Optional, NoReturn, Callable
import code
import curtsies
import cwcwidth
import greenlet
import importlib.util
import logging
import os
import pygments
import requests
import sys
import xdg
from pathlib import Path
from . import __version__, __copyright__
from .config import default_config_path, Config
from .translations import _
def exec_code(interpreter: code.InteractiveInterpreter, args: List[str]) -> None:
    """
    Helper to execute code in a given interpreter, e.g. to implement the behavior of python3 [-i] file.py

    args should be a [faked] sys.argv.
    """
    try:
        with open(args[0]) as sourcefile:
            source = sourcefile.read()
    except OSError as e:
        print(f"bpython: can't open file '{args[0]}: {e}", file=sys.stderr)
        raise SystemExit(e.errno)
    old_argv, sys.argv = (sys.argv, args)
    sys.path.insert(0, os.path.abspath(os.path.dirname(args[0])))
    spec = importlib.util.spec_from_loader('__main__', loader=None)
    assert spec
    mod = importlib.util.module_from_spec(spec)
    sys.modules['__main__'] = mod
    interpreter.locals.update(mod.__dict__)
    interpreter.locals['__file__'] = args[0]
    interpreter.runsource(source, args[0], 'exec')
    sys.argv = old_argv