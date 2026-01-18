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
def scandir(path: Union[str, 'os.PathLike[str]'], sort_key: Callable[['os.DirEntry[str]'], object]=lambda entry: entry.name) -> List['os.DirEntry[str]']:
    """Scan a directory recursively, in breadth-first order.

    The returned entries are sorted according to the given key.
    The default is to sort by name.
    """
    entries = []
    with os.scandir(path) as s:
        for entry in s:
            try:
                entry.is_file()
            except OSError as err:
                if _ignore_error(err):
                    continue
                raise
            entries.append(entry)
    entries.sort(key=sort_key)
    return entries