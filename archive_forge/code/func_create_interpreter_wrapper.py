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
def create_interpreter_wrapper(interpreter: str, injected_interpreter: str) -> None:
    """Create a wrapper for the given Python interpreter at the specified path."""
    shebang_interpreter = sys.executable
    code = textwrap.dedent("\n    #!%s\n\n    from __future__ import absolute_import\n\n    from os import execv\n    from sys import argv\n\n    python = '%s'\n\n    execv(python, [python] + argv[1:])\n    " % (shebang_interpreter, interpreter)).lstrip()
    write_text_file(injected_interpreter, code)
    verified_chmod(injected_interpreter, MODE_FILE_EXECUTE)