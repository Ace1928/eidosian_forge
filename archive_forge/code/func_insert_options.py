from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import tempfile
import typing as t
from .constants import (
from .locale_util import (
from .io import (
from .config import (
from .util import (
from .util_common import (
from .ansible_util import (
from .containers import (
from .data import (
from .payload import (
from .ci import (
from .host_configs import (
from .connections import (
from .provisioning import (
from .content_config import (
def insert_options(command: list[str], options: list[str]) -> list[str]:
    """Insert addition command line options into the given command and return the result."""
    result = []
    for arg in command:
        if options and arg.startswith('--'):
            result.extend(options)
            options = None
        result.append(arg)
    return result