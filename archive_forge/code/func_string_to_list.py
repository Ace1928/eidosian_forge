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
def string_to_list(value: str, intify: bool=False) -> Union[List[str], List[int]]:
    """Parse a comma-separated string to a list and account for various
    formatting options. Mostly used to handle CLI arguments that take a list of
    comma-separated values.

    value (str): The value to parse.
    intify (bool): Whether to convert values to ints.
    RETURNS (Union[List[str], List[int]]): A list of strings or ints.
    """
    if not value:
        return []
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    result = []
    for p in value.split(','):
        p = p.strip()
        if p.startswith("'") and p.endswith("'"):
            p = p[1:-1]
        if p.startswith('"') and p.endswith('"'):
            p = p[1:-1]
        p = p.strip()
        if intify:
            p = int(p)
        result.append(p)
    return result