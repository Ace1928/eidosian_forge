import asyncio
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Union
from unittest import mock
import torch
import accelerate
from ..state import AcceleratorState, PartialState
from ..utils import (
def run_command(command: List[str], return_stdout=False, env=None):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occured while running `command`
    """
    for i, c in enumerate(command):
        if isinstance(c, Path):
            command[i] = str(c)
    if env is None:
        env = os.environ.copy()
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)
        if return_stdout:
            if hasattr(output, 'decode'):
                output = output.decode('utf-8')
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(f'Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}') from e