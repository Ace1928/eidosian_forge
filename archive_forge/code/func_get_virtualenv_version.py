from __future__ import annotations
import collections.abc as c
import json
import os
import pathlib
import sys
import typing as t
from .config import (
from .util import (
from .util_common import (
from .host_configs import (
from .python_requirements import (
def get_virtualenv_version(args: EnvironmentConfig, python: str) -> t.Optional[tuple[int, ...]]:
    """Get the virtualenv version for the given python interpreter, if available, otherwise return None."""
    try:
        cache = get_virtualenv_version.cache
    except AttributeError:
        cache = get_virtualenv_version.cache = {}
    if python not in cache:
        try:
            stdout = run_command(args, [python, '-m', 'virtualenv', '--version'], capture=True)[0]
        except SubprocessError as ex:
            stdout = ''
            if args.verbosity > 1:
                display.error(ex.message)
        version = None
        if stdout:
            try:
                version = str_to_version(stdout.strip())
            except Exception:
                pass
        cache[python] = version
    version = cache[python]
    return version