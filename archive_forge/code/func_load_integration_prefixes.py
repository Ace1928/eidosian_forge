from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
def load_integration_prefixes() -> dict[str, str]:
    """Load and return the integration test prefixes."""
    path = data_context().content.integration_path
    file_paths = sorted((f for f in data_context().content.get_files(path) if os.path.splitext(os.path.basename(f))[0] == 'target-prefixes'))
    prefixes = {}
    for file_path in file_paths:
        prefix = os.path.splitext(file_path)[1][1:]
        prefixes.update(dict(((k, prefix) for k in read_text_file(file_path).splitlines())))
    return prefixes