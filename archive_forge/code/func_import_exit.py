import argparse
import logging
import os
import shlex
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, List, Optional, Tuple, Union, cast
from . import Change
from .filters import BaseFilter, DefaultFilter, PythonFilter
from .run import detect_target_type, import_string, run_process
from .version import VERSION
def import_exit(function_path: str) -> Any:
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)
    try:
        return import_string(function_path)
    except ImportError as e:
        print(f'ImportError: {e}', file=sys.stderr)
        sys.exit(1)