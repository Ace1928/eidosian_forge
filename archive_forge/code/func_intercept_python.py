from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def intercept_python(args: CommonConfig, python: PythonConfig, cmd: list[str], env: dict[str, str], capture: bool, data: t.Optional[str]=None, cwd: t.Optional[str]=None, always: bool=False) -> tuple[t.Optional[str], t.Optional[str]]:
    """
    Run a command while intercepting invocations of Python to control the version used.
    If the specified Python is an ansible-test managed virtual environment, it will be added to PATH to activate it.
    Otherwise a temporary directory will be created to ensure the correct Python can be found in PATH.
    """
    env = env.copy()
    cmd = list(cmd)
    inject_path = get_injector_path()
    if isinstance(python, VirtualPythonConfig):
        python_path = os.path.dirname(python.path)
    else:
        python_path = get_python_path(python.path)
    env['PATH'] = os.path.pathsep.join([inject_path, python_path, env['PATH']])
    env['ANSIBLE_TEST_PYTHON_VERSION'] = python.version
    env['ANSIBLE_TEST_PYTHON_INTERPRETER'] = python.path
    return run_command(args, cmd, capture=capture, env=env, data=data, cwd=cwd, always=always)