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
def create_temp_dir(prefix: t.Optional[str]=None, suffix: t.Optional[str]=None, base_dir: t.Optional[str]=None) -> str:
    """Create a temporary directory that persists until the current process exits."""
    temp_path = tempfile.mkdtemp(prefix=prefix or 'tmp', suffix=suffix or '', dir=base_dir)
    ExitHandler.register(remove_tree, temp_path)
    return temp_path