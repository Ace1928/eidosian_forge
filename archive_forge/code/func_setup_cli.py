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
def setup_cli() -> None:
    registry.cli.get_all()
    command = get_command(app)
    command(prog_name=COMMAND)