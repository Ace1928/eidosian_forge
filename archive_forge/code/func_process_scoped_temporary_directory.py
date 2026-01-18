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
def process_scoped_temporary_directory(args: CommonConfig, prefix: t.Optional[str]='ansible-test-', suffix: t.Optional[str]=None) -> str:
    """Return the path to a temporary directory that will be automatically removed when the process exits."""
    if args.explain:
        path = os.path.join(tempfile.gettempdir(), f'{prefix or tempfile.gettempprefix()}{generate_name()}{suffix or ''}')
    else:
        path = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
        ExitHandler.register(lambda: remove_tree(path))
    return path