import atexit
import contextlib
from enum import Enum
from errno import EBADF
from errno import ELOOP
from errno import ENOENT
from errno import ENOTDIR
import fnmatch
from functools import partial
import importlib.util
import itertools
import os
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import sep
from pathlib import Path
from pathlib import PurePath
from posixpath import sep as posix_sep
import shutil
import sys
import types
from types import ModuleType
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import uuid
import warnings
from _pytest.compat import assert_never
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
def cleanup_candidates(root: Path, prefix: str, keep: int) -> Iterator[Path]:
    """List candidates for numbered directories to be removed - follows py.path."""
    max_existing = max(map(parse_num, find_suffixes(root, prefix)), default=-1)
    max_delete = max_existing - keep
    entries = find_prefixed(root, prefix)
    entries, entries2 = itertools.tee(entries)
    numbers = map(parse_num, extract_suffixes(entries2, prefix))
    for entry, number in zip(entries, numbers):
        if number <= max_delete:
            yield Path(entry)