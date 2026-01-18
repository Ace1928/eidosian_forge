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
def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    else:
        return path.resolve()