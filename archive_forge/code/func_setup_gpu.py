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
def setup_gpu(use_gpu: int, silent=None) -> None:
    """Configure the GPU and log info."""
    if silent is None:
        local_msg = Printer()
    else:
        local_msg = Printer(no_print=silent, pretty=not silent)
    if use_gpu >= 0:
        local_msg.info(f'Using GPU: {use_gpu}')
        require_gpu(use_gpu)
    else:
        local_msg.info('Using CPU')
        if gpu_is_available():
            local_msg.info('To switch to GPU 0, use the option: --gpu-id 0')