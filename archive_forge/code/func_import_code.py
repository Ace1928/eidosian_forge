import hashlib
import os
import shutil
import sys
from configparser import InterpolationError
from contextlib import contextmanager
from pathlib import Path
from typing import (
import srsly
import typer
from click import NoSuchOption
from click.parser import split_arg_string
from thinc.api import Config, ConfigValidationError, require_gpu
from thinc.util import gpu_is_available
from typer.main import get_command
from wasabi import Printer, msg
from weasel import app as project_cli
from .. import about
from ..compat import Literal
from ..schemas import validate
from ..util import (
def import_code(code_path: Optional[Union[Path, str]]) -> None:
    """Helper to import Python file provided in training commands / commands
    using the config. This makes custom registered functions available.
    """
    if code_path is not None:
        if not Path(code_path).exists():
            msg.fail('Path to Python code not found', code_path, exits=1)
        try:
            import_file('python_code', code_path)
        except Exception as e:
            msg.fail(f"Couldn't load Python code: {code_path}", e, exits=1)