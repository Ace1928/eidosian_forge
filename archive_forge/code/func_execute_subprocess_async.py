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
def execute_subprocess_async(cmd: list, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    for i, c in enumerate(cmd):
        if isinstance(c, Path):
            cmd[i] = str(c)
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo))
    cmd_str = ' '.join(cmd)
    if result.returncode > 0:
        stderr = '\n'.join(result.stderr)
        raise RuntimeError(f"'{cmd_str}' failed with returncode {result.returncode}\n\nThe combined stderr from workers follows:\n{stderr}")
    return result