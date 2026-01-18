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
def set_shebang(script: str, executable: str) -> str:
    """Return the given script with the specified executable used for the shebang."""
    prefix = '#!'
    shebang = prefix + executable
    overwrite = (prefix, '# auto-shebang', '# shellcheck shell=')
    lines = script.splitlines()
    if any((lines[0].startswith(value) for value in overwrite)):
        lines[0] = shebang
    else:
        lines.insert(0, shebang)
    script = '\n'.join(lines)
    return script